"""Shared, read-only Lift-cube grasp telemetry for teleoperation entrypoints.

This module does not create a controller, support, weld, or physics shortcut.
It only reads the current MuJoCo state after the caller has stepped the
environment and reports a common interactive-grasp criterion:

* cube rises at least the configured height from its reset position;
* it stays off the table while in robot-hand contact for the configured time;
* cube contact penetration stays below the configured diagnostic limit.

The result is evidence for a human-operated attempt, not an automatic grasp
planner or a substitute for Gate 2's separate static-grasp protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import mujoco
except ModuleNotFoundError:  # pragma: no cover - project lock supplies MuJoCo
    mujoco = None  # type: ignore[assignment]


def _native(value: Any) -> Any:
    return getattr(value, "_model", getattr(value, "_data", value))


@dataclass(frozen=True)
class GraspTelemetryConfig:
    """One common, conservative interactive-grasp observation threshold set."""

    lift_height_m: float = 0.03
    hold_duration_s: float = 3.0
    max_penetration_m: float = 0.004
    cube_body_name: str = "cube_main"
    table_geom_name: str = "table_collision"

    def __post_init__(self) -> None:
        for field in ("lift_height_m", "hold_duration_s", "max_penetration_m"):
            value = float(getattr(self, field))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"grasp_telemetry.{field} must be finite and > 0")
            object.__setattr__(self, field, value)
        for field in ("cube_body_name", "table_geom_name"):
            value = str(getattr(self, field)).strip()
            if not value or "\x00" in value:
                raise ValueError(f"grasp_telemetry.{field} must be nonempty text")
            object.__setattr__(self, field, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraspTelemetry:
    """Read Lift cube motion and contacts without changing the MuJoCo model."""

    def __init__(self, env: Any, config: GraspTelemetryConfig):
        if not isinstance(config, GraspTelemetryConfig):
            raise TypeError("config must be a validated GraspTelemetryConfig")
        self.env = env
        self.config = config
        self.sim = getattr(env, "sim", env)
        self.model = getattr(self.sim, "model", None)
        self.data = getattr(self.sim, "data", None)
        self.native_model = _native(self.model) if self.model is not None else None
        self.native_data = _native(self.data) if self.data is not None else None
        self._cube_body_id = -1
        self._cube_geom_ids: set[int] = set()
        self._table_geom_ids: set[int] = set()
        self._initial_z: float | None = None
        self._hold_started_s: float | None = None
        self._hold_duration_s = 0.0
        self._maximum_lift_m = 0.0
        self._maximum_penetration_m = 0.0
        self._passed = False
        self._unavailable_reason: str | None = None
        self._initialize()

    @property
    def available(self) -> bool:
        return self._unavailable_reason is None

    def _name(self, kind: Any, ident: int) -> str | None:
        if mujoco is None or self.native_model is None:
            return None
        value = mujoco.mj_id2name(self.native_model, kind, int(ident))
        return None if value is None else str(value)

    def _find_body(self, name: str) -> int:
        if mujoco is None or self.native_model is None:
            return -1
        exact = int(mujoco.mj_name2id(self.native_model, mujoco.mjtObj.mjOBJ_BODY, name))
        if exact >= 0:
            return exact
        matches = [
            ident for ident in range(int(self.native_model.nbody))
            if (candidate := self._name(mujoco.mjtObj.mjOBJ_BODY, ident)) is not None
            and candidate.endswith(f"_{name}")
        ]
        return matches[0] if len(matches) == 1 else -1

    def _belongs_to_cube(self, body_id: int) -> bool:
        if self.model is None or self._cube_body_id < 0:
            return False
        current = int(body_id)
        while current not in (0, -1):
            if current == self._cube_body_id:
                return True
            current = int(self.model.body_parentid[current])
        return False

    def _initialize(self) -> None:
        if mujoco is None or self.model is None or self.data is None or self.native_model is None:
            self._unavailable_reason = "mujoco_sim_unavailable"
            return
        self._cube_body_id = self._find_body(self.config.cube_body_name)
        if self._cube_body_id < 0:
            self._unavailable_reason = f"missing_cube_body:{self.config.cube_body_name}"
            return
        self._cube_geom_ids = {
            ident for ident in range(int(self.native_model.ngeom))
            if self._belongs_to_cube(int(self.model.geom_bodyid[ident]))
        }
        self._table_geom_ids = {
            ident for ident in range(int(self.native_model.ngeom))
            if self._name(mujoco.mjtObj.mjOBJ_GEOM, ident) == self.config.table_geom_name
        }
        if not self._cube_geom_ids:
            self._unavailable_reason = "cube_has_no_geometries"
            return
        if not self._table_geom_ids:
            self._unavailable_reason = f"missing_table_geom:{self.config.table_geom_name}"
            return
        self._initial_z = float(self.data.xpos[self._cube_body_id][2])

    def _contact_snapshot(self) -> tuple[bool, set[str], float]:
        table_contact = False
        hand_geoms: set[str] = set()
        maximum_penetration = 0.0
        if self.data is None or self.native_model is None or mujoco is None:
            return table_contact, hand_geoms, maximum_penetration
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in self._cube_geom_ids and second not in self._cube_geom_ids:
                continue
            other = second if first in self._cube_geom_ids else first
            if other in self._table_geom_ids:
                table_contact = True
            name = self._name(mujoco.mjtObj.mjOBJ_GEOM, other)
            if name and (name.startswith("robot0_") or name.startswith("gripper0_")):
                hand_geoms.add(name)
            distance = float(contact.dist)
            if np.isfinite(distance):
                maximum_penetration = max(maximum_penetration, -distance)
        return table_contact, hand_geoms, maximum_penetration

    def update(self) -> dict[str, Any]:
        """Read one already-stepped scene and return JSON-ready telemetry."""
        if not self.available or self.data is None or self._initial_z is None:
            return {
                "available": False,
                "reason": self._unavailable_reason or "uninitialized",
                "passed": False,
            }
        sim_time = float(self.data.time)
        cube_z = float(self.data.xpos[self._cube_body_id][2])
        lift = max(0.0, cube_z - self._initial_z)
        table_contact, hand_geoms, penetration = self._contact_snapshot()
        self._maximum_lift_m = max(self._maximum_lift_m, lift)
        self._maximum_penetration_m = max(self._maximum_penetration_m, penetration)
        support_free = not table_contact
        condition = (
            lift >= self.config.lift_height_m
            and support_free
            and bool(hand_geoms)
            and penetration <= self.config.max_penetration_m
        )
        if condition:
            if self._hold_started_s is None:
                self._hold_started_s = sim_time
            self._hold_duration_s = max(0.0, sim_time - self._hold_started_s)
        else:
            self._hold_started_s = None
            self._hold_duration_s = 0.0
        if self._hold_duration_s >= self.config.hold_duration_s:
            self._passed = True
        return {
            "available": True,
            "sim_time_s": sim_time,
            "cube_height_m": cube_z,
            "lift_height_m": lift,
            "maximum_lift_m": self._maximum_lift_m,
            "table_contact": table_contact,
            "support_free": support_free,
            "hand_contact_geoms": sorted(hand_geoms),
            "hand_contact": bool(hand_geoms),
            "penetration_m": penetration,
            "maximum_penetration_m": self._maximum_penetration_m,
            "hold_duration_s": self._hold_duration_s,
            "required_hold_duration_s": self.config.hold_duration_s,
            "passed": self._passed,
            "criteria_met_this_step": condition,
        }

    def summary(self) -> dict[str, Any]:
        """Return a final deterministic result without modifying simulation."""
        if not self.available:
            return {"available": False, "reason": self._unavailable_reason, "passed": False}
        if self._passed:
            failure = None
        elif self._maximum_penetration_m > self.config.max_penetration_m:
            failure = "cube_penetration_limit"
        elif self._maximum_lift_m < self.config.lift_height_m:
            failure = "cube_not_lifted"
        elif self._hold_duration_s <= 0.0:
            failure = "not_support_free_hand_hold"
        else:
            failure = "hold_duration_incomplete"
        return {
            "available": True,
            "passed": self._passed,
            "failure_reason": failure,
            "maximum_lift_m": self._maximum_lift_m,
            "maximum_penetration_m": self._maximum_penetration_m,
            "last_hold_duration_s": self._hold_duration_s,
            "criteria": self.config.to_dict(),
            "physics_modification": "none_read_only",
        }
