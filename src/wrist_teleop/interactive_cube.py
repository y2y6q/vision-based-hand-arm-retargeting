"""Small, explicit runtime scaling helper for the interactive Lift cube.

This helper is intentionally limited to the Panda + Allegro interactive
entrypoint.  It changes the compiled Lift cube's collision and visual box
geometries together, preserves the original material density by updating the
free body's mass and inertia, and raises the free joint enough to keep the
bottom face from suddenly penetrating the table.  It is never used by Gate 2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # The locked robosuite environment includes MuJoCo; keep imports safe for docs/tests.
    import mujoco
except ModuleNotFoundError:  # pragma: no cover - exercised only without the locked environment
    mujoco = None  # type: ignore[assignment]


class InteractiveCubeScaler:
    """Idempotently enlarge Lift's existing free cube while keeping physics coherent."""

    def __init__(
        self,
        env: Any,
        *,
        factor: float = 1.5,
        body_name: str = "cube_main",
        joint_name: str = "cube_joint0",
    ) -> None:
        self.env = env
        self.factor = float(factor)
        self.body_name = str(body_name)
        self.joint_name = str(joint_name)
        self.sim = getattr(env, "sim", env)
        self.model = getattr(getattr(self.sim, "model", None), "_model", getattr(self.sim, "model", None))
        self.data = getattr(getattr(self.sim, "data", None), "_data", getattr(self.sim, "data", None))
        self._body_id = -1
        self._joint_id = -1
        self._geom_ids: tuple[int, ...] = ()
        self._original_sizes: np.ndarray | None = None
        self._original_mass: float | None = None
        self._original_inertia: np.ndarray | None = None
        self._current_factor = 1.0
        self._reason: str | None = None
        self._initialize()

    @property
    def available(self) -> bool:
        return self._reason is None

    @property
    def enlarged(self) -> bool:
        return self.available and self._current_factor > 1.0 + 1.0e-12

    def _initialize(self) -> None:
        if not np.isfinite(self.factor) or self.factor <= 1.0:
            self._reason = "factor_must_be_finite_and_greater_than_one"
            return
        if mujoco is None or self.model is None or self.data is None:
            self._reason = "mujoco_sim_unavailable"
            return
        self._body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name))
        self._joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name))
        if self._body_id < 0:
            self._reason = f"missing_cube_body:{self.body_name}"
            return
        if self._joint_id < 0:
            self._reason = f"missing_cube_joint:{self.joint_name}"
            return
        if int(self.model.jnt_bodyid[self._joint_id]) != self._body_id:
            self._reason = "cube_joint_not_owned_by_cube_body"
            return
        if int(self.model.jnt_type[self._joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            self._reason = "cube_joint_is_not_free"
            return
        geom_ids = tuple(
            int(geom_id)
            for geom_id in range(int(self.model.ngeom))
            if int(self.model.geom_bodyid[geom_id]) == self._body_id
        )
        if not geom_ids:
            self._reason = "cube_has_no_direct_geometries"
            return
        if any(int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX) for geom_id in geom_ids):
            self._reason = "cube_geometry_is_not_all_boxes"
            return
        sizes = np.asarray([self.model.geom_size[geom_id] for geom_id in geom_ids], dtype=np.float64)
        if sizes.ndim != 2 or sizes.shape[1] < 3 or not np.all(np.isfinite(sizes[:, :3])) or np.any(sizes[:, :3] <= 0.0):
            self._reason = "cube_has_invalid_box_sizes"
            return
        mass = float(self.model.body_mass[self._body_id])
        inertia = np.asarray(self.model.body_inertia[self._body_id], dtype=np.float64).copy()
        if not np.isfinite(mass) or mass <= 0.0 or inertia.shape != (3,) or not np.all(np.isfinite(inertia)) or np.any(inertia <= 0.0):
            self._reason = "cube_has_invalid_inertial_properties"
            return
        self._geom_ids = geom_ids
        self._original_sizes = sizes[:, :3].copy()
        self._original_mass = mass
        self._original_inertia = inertia

    def status(self) -> dict[str, Any]:
        """Return only facts needed by the entrypoint's JSON log."""
        record: dict[str, Any] = {
            "available": self.available,
            "cube_body": self.body_name,
            "cube_joint": self.joint_name,
            "current_factor": float(self._current_factor),
            "enlarged": self.enlarged,
            "requested_factor": float(self.factor),
        }
        if self._reason is not None:
            record["reason"] = self._reason
        if self._original_sizes is not None:
            record["default_half_sizes_m"] = self._original_sizes.tolist()
        if self._original_mass is not None:
            record["default_mass_kg"] = self._original_mass
        if self._original_inertia is not None:
            record["default_inertia_kg_m2"] = self._original_inertia.tolist()
        return record

    def _bottom_z(self, half_sizes: np.ndarray) -> float:
        assert self.data is not None
        bottoms: list[float] = []
        for geom_id, size in zip(self._geom_ids, half_sizes, strict=True):
            center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
            rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            vertical_half_extent = float(np.abs(rotation[2, :]) @ np.asarray(size, dtype=np.float64))
            bottoms.append(float(center[2] - vertical_half_extent))
        return min(bottoms)

    def _capture_dynamic_state(self) -> dict[str, Any]:
        """Keep the current scene intact across MuJoCo's constant rebuild."""
        assert self.data is not None
        state: dict[str, Any] = {"time": float(self.data.time)}
        for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata"):
            values = getattr(self.data, field, None)
            if values is not None:
                state[field] = np.asarray(values, dtype=np.float64).copy()
        return state

    def _restore_dynamic_state(self, state: dict[str, Any]) -> None:
        assert self.data is not None
        for field, values in state.items():
            if field == "time":
                self.data.time = float(values)
                continue
            target = getattr(self.data, field, None)
            if target is not None and np.shape(target) == np.shape(values):
                target[...] = values

    def _refresh_constants(self, state: dict[str, Any]) -> None:
        assert mujoco is not None and self.model is not None and self.data is not None
        # MuJoCo deliberately does not infer body inertias from edited geom
        # sizes at runtime.  The caller updates them first, then this rebuilds
        # constants (including subtree mass) before a normal forward pass.
        # ``mj_setConst`` resets the supplied MjData, so restore every dynamic
        # state vector before forwarding.  Resizing the cube must not reset the
        # Panda, Allegro, time, targets, or free-body velocity.
        mujoco.mj_setConst(self.model, self.data)
        self._restore_dynamic_state(state)
        mujoco.mj_forward(self.model, self.data)

    def enlarge(self) -> dict[str, Any]:
        """Apply the configured enlargement once; later key repeats are harmless."""
        if not self.available:
            return {**self.status(), "changed": False}
        if self.enlarged:
            return {**self.status(), "changed": False, "reason": "already_enlarged"}
        assert self.model is not None and self.data is not None
        assert self._original_sizes is not None and self._original_mass is not None and self._original_inertia is not None
        before_sizes = np.asarray([self.model.geom_size[geom_id][:3] for geom_id in self._geom_ids], dtype=np.float64)
        before_bottom_z = self._bottom_z(before_sizes)
        new_sizes = self._original_sizes * self.factor
        after_bottom_z = self._bottom_z(new_sizes)
        upward_shift = max(0.0, before_bottom_z - after_bottom_z)
        qpos_adr = int(self.model.jnt_qposadr[self._joint_id])
        prior_state = self._capture_dynamic_state()
        before_qpos = np.asarray(prior_state["qpos"][qpos_adr:qpos_adr + 7], dtype=np.float64).copy()
        qvel_adr = int(self.model.jnt_dofadr[self._joint_id])
        before_qvel = np.asarray(prior_state["qvel"][qvel_adr:qvel_adr + 6], dtype=np.float64).copy()
        enlarged_state = dict(prior_state)
        enlarged_qpos = np.asarray(prior_state["qpos"], dtype=np.float64).copy()
        enlarged_qpos[qpos_adr + 2] += upward_shift
        enlarged_state["qpos"] = enlarged_qpos
        try:
            for geom_id, size in zip(self._geom_ids, new_sizes, strict=True):
                self.model.geom_size[geom_id][:3] = size
            self.model.body_mass[self._body_id] = self._original_mass * self.factor ** 3
            self.model.body_inertia[self._body_id] = self._original_inertia * self.factor ** 5
            # Preserve the current free-joint velocity.  Only a positive world-Z
            # translation is made, preventing a larger cube from appearing
            # through the table at the moment the user presses B.
            self._refresh_constants(enlarged_state)
        except Exception as exc:
            for geom_id, size in zip(self._geom_ids, before_sizes, strict=True):
                self.model.geom_size[geom_id][:3] = size
            self.model.body_mass[self._body_id] = self._original_mass
            self.model.body_inertia[self._body_id] = self._original_inertia
            self._refresh_constants(prior_state)
            return {**self.status(), "changed": False, "reason": f"cube_size_update_failed:{type(exc).__name__}:{exc}"}
        self._current_factor = self.factor
        after_qpos = np.asarray(self.data.qpos[qpos_adr:qpos_adr + 7], dtype=np.float64).copy()
        after_qvel = np.asarray(self.data.qvel[qvel_adr:qvel_adr + 6], dtype=np.float64).copy()
        return {
            **self.status(),
            "changed": True,
            "event": "cube_size_changed",
            "old_half_sizes_m": before_sizes.tolist(),
            "new_half_sizes_m": new_sizes.tolist(),
            "mass_before_kg": self._original_mass,
            "mass_after_kg": float(self.model.body_mass[self._body_id]),
            "inertia_before_kg_m2": self._original_inertia.tolist(),
            "inertia_after_kg_m2": np.asarray(self.model.body_inertia[self._body_id], dtype=np.float64).tolist(),
            "free_joint_qpos_before": before_qpos.tolist(),
            "free_joint_qpos_after": after_qpos.tolist(),
            "position_shift_world_m": [0.0, 0.0, float(upward_shift)],
            "velocity_preserved": bool(np.array_equal(before_qvel, after_qvel)),
            "physics_refresh": "mj_setConst+mj_forward",
        }

    def restore_default(self) -> dict[str, Any]:
        """Restore model sizes / inertia before an R-triggered environment reset."""
        if not self.available or not self.enlarged:
            return {**self.status(), "changed": False}
        assert self.model is not None and self.data is not None
        assert self._original_sizes is not None and self._original_mass is not None and self._original_inertia is not None
        for geom_id, size in zip(self._geom_ids, self._original_sizes, strict=True):
            self.model.geom_size[geom_id][:3] = size
        self.model.body_mass[self._body_id] = self._original_mass
        self.model.body_inertia[self._body_id] = self._original_inertia
        self._refresh_constants(self._capture_dynamic_state())
        self._current_factor = 1.0
        return {**self.status(), "changed": True, "event": "cube_size_restored_before_reset"}
