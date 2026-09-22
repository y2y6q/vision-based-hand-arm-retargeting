"""No-simulation availability audit for the project-local Shadow Hand adapter.

The runnable Panda + Shadow Hand path is implemented by
``wrist_teleop.shadow_gripper.ShadowHandRight``. This module deliberately
performs only static checks and a robosuite factory registration: it never
constructs a MuJoCo model, opens a HID device, or starts a camera. The audit
therefore remains safe to use as an installation and asset preflight.

The adapter uses the official MuJoCo Menagerie E3M5 right hand asset. Its
physical model has 24 hinge joints, while its official action contract has 20
position actuators because four distal-joint pairs are tendon-coupled. It is
not an Allegro substitution. dex-retargeting is intentionally *not*
integrated in this Shadow entry yet; the runnable hand input path is the
project's Shadow-specific legacy curl / safe-open controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SHADOW_TOKEN = "shadow"
_OPTIONAL_MODEL_PACKAGES = ("robosuite_models", "mujoco_menagerie", "dm_control")
OFFICIAL_MENAGERIE_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
OFFICIAL_MENAGERIE_COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
OFFICIAL_MENAGERIE_MODEL_PATH = "shadow_hand/right_hand.xml"
OFFICIAL_MENAGERIE_LICENSE = "Apache-2.0"
_LOCAL_MODEL_CANDIDATES = (
    REPOSITORY_ROOT / "assets" / "shadow_hand" / "right_hand.xml",
    REPOSITORY_ROOT / "assets" / "shadow_hand" / "right_hand_flattened.xml",
    REPOSITORY_ROOT / "assets" / "shadow_hand" / "shadow_hand.xml",
    REPOSITORY_ROOT / "models" / "shadow_hand" / "shadow_hand.xml",
    REPOSITORY_ROOT / "models" / "shadow_hand" / "right_hand.xml",
)


@dataclass(frozen=True)
class ShadowHandAudit:
    """JSON-safe facts for a Shadow Hand preflight, without simulation."""

    robosuite_version: str
    registered_grippers: tuple[str, ...]
    shadow_gripper_names: tuple[str, ...]
    optional_model_packages: dict[str, bool]
    local_model_candidates: tuple[str, ...]
    existing_local_model_candidates: tuple[str, ...]
    official_model_reference: dict[str, Any]
    adapter: dict[str, Any]
    physical_joint_count: int
    action_dof: int
    dex_not_integrated: bool
    ready: bool
    required_integration_contract: tuple[str, ...]
    blocker: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShadowHandUnavailable(RuntimeError):
    """Raised before startup when the local Shadow runtime preflight fails."""


def _robosuite_version() -> str:
    try:
        return importlib.metadata.version("robosuite")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _mesh_manifest(source: Path) -> tuple[bool, tuple[str, ...]]:
    """Return whether every mesh referenced by a Menagerie source exists."""

    if not source.is_file():
        return False, ()
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError:
        return False, ()
    mesh_root = source.parent / "assets"
    missing: list[str] = []
    for mesh in root.findall("./asset/mesh"):
        raw = mesh.get("file")
        if raw and not (mesh_root / raw).is_file():
            missing.append(raw)
    return not missing, tuple(missing)


def _adapter_facts() -> tuple[dict[str, Any], type[Any] | None, str | None]:
    """Import only the project adapter and inspect its declared contract."""

    try:
        from .shadow_gripper import (
            GRIPPER_TYPE,
            SHADOW_ACTUATOR_NAMES,
            SHADOW_JOINT_NAMES,
            SOURCE_MJCF,
            ShadowHandRight,
            register_shadow_gripper,
        )
    except Exception as exc:  # pragma: no cover - exercised on broken installs
        return (
            {
                "module": "wrist_teleop.shadow_gripper",
                "registered_type": None,
                "source_mjcf": None,
                "source_meshes_complete": False,
                "missing_meshes": [],
            },
            None,
            f"cannot import the project-local Shadow adapter: {type(exc).__name__}: {exc}",
        )

    meshes_complete, missing_meshes = _mesh_manifest(SOURCE_MJCF)
    facts = {
        "module": "wrist_teleop.shadow_gripper",
        "class": ShadowHandRight.__name__,
        "registered_type": GRIPPER_TYPE,
        "source_mjcf": str(SOURCE_MJCF),
        "source_meshes_complete": meshes_complete,
        "missing_meshes": list(missing_meshes),
        "physical_joint_order": list(SHADOW_JOINT_NAMES),
        "actuator_order": list(SHADOW_ACTUATOR_NAMES),
        "tendon_coupled_actuators": ["rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"],
        "factory_registration": "project-local register_shadow_gripper()",
    }
    # Registering a Python class is not simulation; it simply makes the
    # project-local gripper visible to robosuite's live factory.
    register_shadow_gripper()
    return facts, ShadowHandRight, None


def audit_shadow_hand_support(
    *,
    gripper_mapping: Mapping[Any, Any] | None = None,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    local_model_candidates: tuple[Path, ...] = _LOCAL_MODEL_CANDIDATES,
) -> ShadowHandAudit:
    """Audit the project-local adapter without starting simulation or hardware.

    With the normal live robosuite mapping, the adapter is registered before
    inspection. Supplying ``gripper_mapping`` keeps the function pure for
    failure-path tests and verifies that mapping exactly as supplied.
    """

    adapter, adapter_class, adapter_error = _adapter_facts()
    if gripper_mapping is None:
        from robosuite.models.grippers import GRIPPER_MAPPING

        gripper_mapping = GRIPPER_MAPPING

    registered = tuple(sorted(str(name) for name in gripper_mapping if name is not None))
    shadow_names = tuple(name for name in registered if _SHADOW_TOKEN in name.lower())
    packages = {name: find_spec(name) is not None for name in _OPTIONAL_MODEL_PACKAGES}
    candidate_paths = tuple(str(path) for path in local_model_candidates)
    existing_paths = tuple(str(path) for path in local_model_candidates if path.is_file())

    expected_type = str(adapter["registered_type"] or "ShadowHandRight")
    factory_registered = adapter_class is not None and gripper_mapping.get(expected_type) is adapter_class
    source_mjcf = Path(str(adapter["source_mjcf"])) if adapter.get("source_mjcf") else None
    source_matches_candidate = source_mjcf is not None and source_mjcf in local_model_candidates
    local_source_present = source_matches_candidate and source_mjcf.is_file()
    meshes_complete = bool(adapter.get("source_meshes_complete"))
    joint_count = len(adapter.get("physical_joint_order", []))
    actuator_count = len(adapter.get("actuator_order", []))
    contract_counts_match = joint_count == 24 and actuator_count == 20

    contract = (
        "project-local ShadowHandRight is registered in robosuite GRIPPER_MAPPING",
        "the local Menagerie MJCF and every referenced mesh are present",
        "the physical contract is 24 hinge joints and the action contract is 20 official position actuators",
        "Panda OSC_POSE uses a six-dimensional arm slice followed by the 20-dimensional Shadow hand slice",
        "dex-retargeting is not integrated for Shadow Hand; use the Shadow legacy-curl / safe-open input path",
    )
    reasons: list[str] = []
    if adapter_error:
        reasons.append(adapter_error)
    if not local_source_present:
        reasons.append("the project-local Shadow MJCF is missing from the supported asset path")
    if not meshes_complete:
        missing = ", ".join(str(value) for value in adapter.get("missing_meshes", [])) or "unknown mesh asset"
        reasons.append(f"the project-local Shadow mesh manifest is incomplete: {missing}")
    if not factory_registered:
        reasons.append(f"robosuite GRIPPER_MAPPING does not contain the registered local type {expected_type!r}")
    if not contract_counts_match:
        reasons.append(f"the adapter contract is {joint_count} physical joints / {actuator_count} actuators, expected 24 / 20")
    blocker = "; ".join(reasons) if reasons else None

    return ShadowHandAudit(
        robosuite_version=_robosuite_version(),
        registered_grippers=registered,
        shadow_gripper_names=shadow_names,
        optional_model_packages=packages,
        local_model_candidates=candidate_paths,
        existing_local_model_candidates=existing_paths,
        official_model_reference={
            "repository": OFFICIAL_MENAGERIE_REPOSITORY,
            "commit": OFFICIAL_MENAGERIE_COMMIT,
            "model_path": OFFICIAL_MENAGERIE_MODEL_PATH,
            "model": "Shadow Hand E3M5 right hand",
            "license": OFFICIAL_MENAGERIE_LICENSE,
            "physical_joint_count": 24,
            "position_actuator_count": 20,
            "action_status": "four distal joint pairs are tendon-coupled; the hand is not a 24-independent-actuator model",
        },
        adapter=adapter,
        physical_joint_count=joint_count,
        action_dof=actuator_count,
        dex_not_integrated=True,
        ready=blocker is None,
        required_integration_contract=contract,
        blocker=blocker,
    )


def require_shadow_hand_support(audit: ShadowHandAudit | None = None) -> ShadowHandAudit:
    """Return a ready audit or raise a concise error before a runtime starts."""

    audit = audit or audit_shadow_hand_support()
    if audit.ready:
        return audit
    raise ShadowHandUnavailable(
        (audit.blocker or "Shadow Hand preflight failed")
        + ". Required runtime contract: "
        + "; ".join(audit.required_integration_contract)
    )
