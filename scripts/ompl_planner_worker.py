"""Approach-planning worker for the LIBERO planner teleop entry.

Runs in the ``ompl_planner`` Conda env (Python 3.11, OMPL 2.0 from
conda-forge, MuJoCo 3.2.3 = the LIBERO env's version). It reads one JSON
request per stdin line and writes one JSON response per stdout line:

- ``init``: load the compiled LIBERO scene (``mj_saveModel`` file) and the
  robot description (arm joints, end-effector site, robot geoms);
- ``plan``: plan from the sent ``qpos`` to the end-effector goal pose;
- ``close``.

One plan, in this order:

1. The lift / traverse / descend candidates the entry sends (hand orientation
   held constant), each checked every ``CARTESIAN_STEP_M`` along the way: IK
   continued from the previous step (a jump to another arm posture rejects
   the candidate, so the hand cannot swing round), joint limits, clearance.
   The first candidate that passes is the plan.
2. Only if all are blocked: OMPL RRT-Connect in the 7 arm joints, with the
   goal IK taken near the current arm posture and the hand orientation kept
   within ``ORIENTATION_TUBE_RAD`` of the goal's along the whole path.

Clearance: MuJoCo ``mj_collision`` with a contact margin on the moving geoms
returns signed distances. Robot + held objects vs. everything else must stay
``TRANSIT_CLEARANCE_M`` apart, and ``APPROACH_CLEARANCE_M`` in the column
right above the goal. Pairs already closer than that at the start (fingers
resting on something, a held object on the table) may not get more than
``START_SLACK_M`` closer. Robot self-contacts are ignored. If the goal itself
is too close to something below it, the plan stops up to ``GOAL_LIFT_MAX_M``
higher. The start joints are clamped into the joint range first (the arm can
sit on a limit; that made every plan fail on the 2026-11-06 real run).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og

TRANSIT_CLEARANCE_M = 0.015
APPROACH_CLEARANCE_M = 0.005
START_SLACK_M = 0.002
CONTACT_MARGIN_M = 0.02  # >= every clearance: contacts are reported up to this distance
GOAL_COLUMN_RADIUS_M = 0.02
GOAL_LIFT_MAX_M = 0.04
IK_POSITION_TOLERANCE_M = 0.002
IK_ROTATION_TOLERANCE_RAD = 0.01
IK_ITERATIONS = 150
STEP_IK_ITERATIONS = 40
JOINT_LIMIT_MARGIN_RAD = 0.005
CARTESIAN_STEP_M = 0.01
CARTESIAN_MAX_JOINT_STEP_RAD = 0.25
ORIENTATION_TUBE_RAD = 0.5
GOAL_IK_SEEDS = 12
GOAL_IK_SEED_SIGMA_RAD = 0.15
GOAL_LIFTS_M = (0.0, 0.01, 0.02, 0.03, 0.04)
RRT_RANGE_RAD = 0.3
VALIDITY_RESOLUTION = 0.004  # fraction of the space extent (~0.04 rad steps here)
WAYPOINT_SPACING_RAD = 0.05


class PlanFailure(RuntimeError):
    """A plan that cannot be made; ``reason`` is one of unreachable / blocked / no_path."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Axis-angle vector of target @ current.T (world frame)."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (target @ current.T).reshape(9))
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(0.5 * (np.trace(a @ b.T) - 1.0), -1.0, 1.0)))


def _densify(points: list[np.ndarray], step: float) -> tuple[list[np.ndarray], list[int]]:
    """Points every ``step`` along the polyline, with the index of the segment each lies on."""
    dense, segment = [points[0]], [0]
    for index in range(1, len(points)):
        a, b = points[index - 1], points[index]
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, count + 1):
            dense.append(a + (b - a) * (k / count))
            segment.append(index - 1)
    return dense, segment


class Planner:
    def __init__(self, request: dict) -> None:
        self.model = mujoco.MjModel.from_binary_path(request["model_path"])
        self.data = mujoco.MjData(self.model)
        self.arm_qpos = np.asarray(request["arm_qpos_adr"], dtype=int)
        self.arm_dof = np.asarray(request["arm_dof_adr"], dtype=int)
        self.site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, request["eef_site"]))
        if self.site < 0:
            raise ValueError("end-effector site not in model: {}".format(request["eef_site"]))
        self.robot_geoms = set(int(g) for g in request["robot_geoms"])
        joints = np.asarray(request["arm_joint_ids"], dtype=int)
        ranges = np.asarray(self.model.jnt_range[joints], dtype=np.float64)
        self.lower = ranges[:, 0] + JOINT_LIMIT_MARGIN_RAD
        self.upper = ranges[:, 1] - JOINT_LIMIT_MARGIN_RAD
        self.base_margin = np.asarray(self.model.geom_margin, dtype=np.float64).copy()
        self.held: list[dict] = []
        self.moving = np.zeros(self.model.ngeom, dtype=bool)
        self.start = np.zeros(len(joints))
        self.start_dist: dict[tuple[int, int], float] = {}
        self.goal_pos = np.zeros(3)
        self.goal_rot = np.eye(3)
        self.checks = 0

    # ------------------------------------------------------------ kinematics
    def _site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def _set_arm(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        if self.held:
            site_pos, site_rot = self._site_pose()
            for held in self.held:
                pos = site_pos + site_rot @ held["rel_pos"]
                rot = site_rot @ held["rel_rot"]
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, rot.reshape(9))
                adr = held["qpos_adr"]
                self.data.qpos[adr:adr + 3] = pos
                self.data.qpos[adr + 3:adr + 7] = quat
            mujoco.mj_kinematics(self.model, self.data)

    def _ik(self, goal_pos: np.ndarray, goal_rot: np.ndarray, seed: np.ndarray,
            iterations: int = IK_ITERATIONS) -> np.ndarray | None:
        seed = np.clip(seed, self.lower, self.upper)
        q = seed.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iterations):
            self.data.qpos[self.arm_qpos] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            pos, rot = self._site_pose()
            error = np.concatenate((goal_pos - pos, _rotation_error(goal_rot, rot)))
            if np.linalg.norm(error[:3]) < IK_POSITION_TOLERANCE_M and np.linalg.norm(error[3:]) < IK_ROTATION_TOLERANCE_RAD:
                return q
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jac = np.vstack((jacp[:, self.arm_dof], jacr[:, self.arm_dof]))
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), error)
            # Small null-space pull toward the seed keeps the elbow where it was.
            null = (np.eye(len(q)) - np.linalg.pinv(jac) @ jac) @ (seed - q)
            q = np.clip(q + np.clip(step, -0.2, 0.2) + 0.05 * null, self.lower, self.upper)
        return None

    # ------------------------------------------------------------- clearance
    def _pair_distances(self) -> dict[tuple[int, int], float]:
        """Signed distance per (moving, static) geom pair closer than the contact margin."""
        mujoco.mj_collision(self.model, self.data)
        count = int(self.data.ncon)
        if count == 0:
            return {}
        g1 = np.asarray(self.data.contact.geom1[:count], dtype=int)
        g2 = np.asarray(self.data.contact.geom2[:count], dtype=int)
        dist = np.asarray(self.data.contact.dist[:count], dtype=np.float64)
        keep = self.moving[g1] != self.moving[g2]
        out: dict[tuple[int, int], float] = {}
        for a, b, d in zip(g1[keep], g2[keep], dist[keep]):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            if d < out.get(key, np.inf):
                out[key] = float(d)
        return out

    def _required(self, ee_pos: np.ndarray) -> float:
        offset = ee_pos - self.goal_pos
        in_column = np.hypot(offset[0], offset[1]) <= GOAL_COLUMN_RADIUS_M and offset[2] >= -0.005
        return APPROACH_CLEARANCE_M if in_column else TRANSIT_CLEARANCE_M

    def _violation(self, ee_pos: np.ndarray,
                   distances: dict[tuple[int, int], float] | None = None) -> tuple[tuple[int, int], float] | None:
        """The first pair closer than allowed at the current state (after ``_set_arm``), else None."""
        required = self._required(ee_pos)
        for key, dist in (self._pair_distances() if distances is None else distances).items():
            start = self.start_dist.get(key)
            limit = required if start is None else min(required, start - START_SLACK_M)
            if dist < limit:
                return key, dist
        return None

    def _name(self, pair: tuple[int, int]) -> str:
        geom = pair[0] if not self.moving[pair[0]] else pair[1]
        body = int(self.model.geom_bodyid[geom])
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or "body {}".format(body)

    def valid(self, q: np.ndarray) -> bool:
        """OMPL state check: joint range, hand orientation tube, clearance."""
        self.checks += 1
        if np.any(q < self.lower - 1e-9) or np.any(q > self.upper + 1e-9):
            return False
        self._set_arm(q)
        pos, rot = self._site_pose()
        if _angle(rot, self.goal_rot) > ORIENTATION_TUBE_RAD:
            return False
        return self._violation(pos) is None

    # ------------------------------------------------------------- strategies
    def _cartesian(self, points: list[np.ndarray]) -> dict:
        """Check one lift / traverse / descend candidate; return the plan or raise PlanFailure."""
        dense, segment = _densify(points, CARTESIAN_STEP_M)
        last = len(points) - 2
        descent_is_vertical = len(points) >= 2 and np.linalg.norm((points[-1] - points[-2])[:2]) < 1e-6
        q = self.start.copy()
        joints = [q.copy()]
        closest = np.inf
        for k in range(1, len(dense)):
            point = dense[k]
            nxt = self._ik(point, self.goal_rot, q, STEP_IK_ITERATIONS)
            if nxt is None:
                raise PlanFailure("unreachable", "out of reach at {} with this wrist orientation".format(
                    np.round(point, 3).tolist()))
            if np.max(np.abs(nxt - q)) > CARTESIAN_MAX_JOINT_STEP_RAD:
                raise PlanFailure("unreachable", "arm posture would jump at {}".format(np.round(point, 3).tolist()))
            self._set_arm(nxt)
            distances = self._pair_distances()
            hit = self._violation(point, distances)
            if hit is not None:
                lift = float(dense[k - 1][2] - points[-1][2])
                if (descent_is_vertical and segment[k] == last and k > 1 and lift <= GOAL_LIFT_MAX_M + 1e-9
                        and np.linalg.norm((dense[k - 1] - points[-1])[:2]) < 1e-6):
                    # Stop the descent above what it would touch.
                    kept = points[:-1] + [dense[k - 1]]
                    if np.linalg.norm(kept[-1] - kept[-2]) < 1e-9:
                        kept = kept[:-1]
                    return {"points": kept, "joints": joints, "goal_lift_m": lift, "closest_m": closest}
                raise PlanFailure("blocked", "would come within {:.0f} mm of {}".format(
                    max(0.0, 1e3 * hit[1]), self._name(hit[0])))
            new_pairs = [d for key, d in distances.items() if key not in self.start_dist]
            if new_pairs:  # closest approach to anything the hand was not already near
                closest = min(closest, min(new_pairs))
            q = nxt
            joints.append(q.copy())
        return {"points": points, "joints": joints, "goal_lift_m": 0.0, "closest_m": closest}

    def _ompl(self, request: dict) -> dict:
        rng = np.random.default_rng(int(request.get("seed", 0)))
        seeds = [self.start] + [np.clip(self.start + rng.normal(0.0, GOAL_IK_SEED_SIGMA_RAD, size=self.start.shape),
                                        self.lower, self.upper) for _ in range(GOAL_IK_SEEDS - 1)]
        best, unreachable, blocked, blocker = None, 0, 0, None
        for lift in GOAL_LIFTS_M:
            goal = self.goal_pos + np.array([0.0, 0.0, lift])
            for seed in seeds:
                q = self._ik(goal, self.goal_rot, seed)
                if q is None:
                    unreachable += 1
                    continue
                if not self.valid(q):
                    blocked += 1
                    hit = self._violation(goal)
                    blocker = self._name(hit[0]) if hit is not None else blocker
                    continue
                distance = float(np.linalg.norm(q - self.start))
                if best is None or distance < best[0]:
                    best = (distance, q, goal, lift)
            if best is not None:
                break
        if best is None:
            if blocked == 0:
                raise PlanFailure("unreachable", "goal out of reach with this wrist orientation (arm joint limits)")
            raise PlanFailure("blocked", "the hand would touch {} at the goal".format(blocker or "something"))
        _distance, goal_q, goal, lift = best
        dimension = len(self.start)
        space = ob.RealVectorStateSpace(dimension)
        bounds = ob.RealVectorBounds(dimension)
        for index in range(dimension):
            bounds.setLow(index, float(self.lower[index]))
            bounds.setHigh(index, float(self.upper[index]))
        space.setBounds(bounds)
        space.setup()
        setup = og.SimpleSetup(space)
        setup.setStateValidityChecker(lambda state: self.valid(np.array([state[i] for i in range(dimension)])))
        setup.getSpaceInformation().setStateValidityCheckingResolution(VALIDITY_RESOLUTION)
        start_state, goal_state = space.allocState(), space.allocState()
        space.copyFromReals(start_state, [float(v) for v in self.start])
        space.copyFromReals(goal_state, [float(v) for v in goal_q])
        setup.setStartAndGoalStates(start_state, goal_state)
        planner = og.RRTConnect(setup.getSpaceInformation())
        planner.setRange(RRT_RANGE_RAD)
        setup.setPlanner(planner)
        time_limit = float(request.get("time_limit_s", 2.0))
        solved = setup.solve(time_limit)
        if not solved or not setup.haveExactSolutionPath():
            raise PlanFailure("no_path", "RRT-Connect found no path in {:.1f} s".format(time_limit))
        setup.simplifySolution(float(request.get("simplify_s", 0.5)))
        path = setup.getSolutionPath()
        joint_length = float(path.length())
        path.interpolate(max(2, int(np.ceil(joint_length / WAYPOINT_SPACING_RAD)) + 1))
        joints, positions, rotations = [], [], []
        for index in range(path.getStateCount()):
            state = path.getState(index)
            q = np.array([state[i] for i in range(dimension)])
            self._set_arm(q)
            pos, rot = self._site_pose()
            joints.append(q.tolist())
            positions.append(pos)
            rotations.append(rot)
        return {"positions": positions, "rotations": rotations, "joints": joints, "goal": goal, "goal_lift_m": lift,
                "joint_path_length_rad": joint_length}

    # ------------------------------------------------------------------ plan
    def plan(self, request: dict) -> dict:
        started = time.perf_counter()
        self.data.qpos[:] = np.asarray(request["qpos"], dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        measured = self.data.qpos[self.arm_qpos].copy()
        self.start = np.clip(measured, self.lower, self.upper)
        self.goal_pos = np.asarray(request["goal_pos"], dtype=np.float64)
        self.goal_rot = np.asarray(request["goal_rot"], dtype=np.float64).reshape(3, 3)
        self._set_arm(self.start)
        site_pos, site_rot = self._site_pose()
        self.held = []
        moving = set(self.robot_geoms)
        for held in request.get("held", []):
            body = int(held["body"])
            joint = int(self.model.body_jntadr[body])
            if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            self.held.append({
                "qpos_adr": int(self.model.jnt_qposadr[joint]),
                "rel_pos": site_rot.T @ (self.data.xpos[body] - site_pos),
                "rel_rot": site_rot.T @ self.data.xmat[body].reshape(3, 3),
            })
            moving.update(int(g) for g in held["geoms"])
        self.moving[:] = False
        self.moving[list(moving)] = True
        self.model.geom_margin[:] = self.base_margin
        self.model.geom_margin[self.moving] = np.maximum(self.base_margin[self.moving], CONTACT_MARGIN_M)
        self._set_arm(self.start)
        self.start_dist = self._pair_distances()
        self.checks = 0
        stats = {"start_clamped_rad": float(np.max(np.abs(measured - self.start))),
                 "start_close_pairs": sum(1 for d in self.start_dist.values() if d < TRANSIT_CLEARANCE_M),
                 "held_objects": len(self.held)}

        failures = []
        for index, candidate in enumerate(request.get("candidates", [])):
            points = [np.asarray(p, dtype=np.float64) for p in candidate]
            try:
                found = self._cartesian(points)
            except PlanFailure as exc:
                failures.append("route {}: {}".format(index + 1, exc))
                continue
            positions = found["points"]
            return {
                "strategy": "cartesian", "segmented": True, "route": index + 1, "routes_tried": index + 1,
                "positions": [p.tolist() for p in positions], "rotations": [self.goal_rot.reshape(9).tolist()] * len(positions),
                "goal_pos": positions[-1].tolist(), "goal_lift_m": found["goal_lift_m"],
                "closest_m": None if not np.isfinite(found["closest_m"]) else found["closest_m"],
                "joints": [q.tolist() for q in found["joints"]], "route_failures": failures,
                "total_ms": 1e3 * (time.perf_counter() - started), **stats,
            }
        try:
            found = self._ompl(request)
        except PlanFailure as exc:
            detail = "; ".join(failures)
            raise PlanFailure(exc.reason, str(exc) + (" (straight routes: {})".format(detail) if detail else "")) from None
        return {
            "strategy": "ompl", "segmented": False, "routes_tried": len(failures),
            "positions": [p.tolist() for p in found["positions"]],
            "rotations": [r.reshape(9).tolist() for r in found["rotations"]],
            "goal_pos": found["goal"].tolist(), "goal_lift_m": found["goal_lift_m"],
            "joints": found["joints"], "joint_path_length_rad": found["joint_path_length_rad"],
            "route_failures": failures, "validity_checks": self.checks,
            "total_ms": 1e3 * (time.perf_counter() - started), **stats,
        }


def main() -> int:
    planner: Planner | None = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        request_id, op = request.get("id"), request.get("op")
        try:
            if op == "init":
                planner = Planner(request)
                result = {"nq": int(planner.model.nq), "ngeom": int(planner.model.ngeom),
                          "mujoco": mujoco.__version__}
            elif op == "plan":
                if planner is None:
                    raise RuntimeError("init first")
                result = planner.plan(request)
            elif op == "close":
                print(json.dumps({"id": request_id, "ok": True, "result": {}}), flush=True)
                break
            else:
                raise ValueError("unknown op {!r}".format(op))
            print(json.dumps({"id": request_id, "ok": True, "result": result}), flush=True)
        except PlanFailure as exc:
            print(json.dumps({"id": request_id, "ok": False, "error": str(exc), "reason": exc.reason}), flush=True)
        except Exception as exc:
            print(json.dumps({"id": request_id, "ok": False,
                              "error": "{}: {}".format(type(exc).__name__, exc),
                              "trace": traceback.format_exc(limit=3)}), flush=True)
    sys.stdout.flush()
    # Skip interpreter teardown: the nanobind bindings print leak reports at exit.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
