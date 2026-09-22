from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.teleop.constrained_ik_controller import (  # noqa: E402
    ConstrainedIKController,
    orientation_error_degrees,
)
from src.teleop.teleop_factory import (  # noqa: E402
    create_environment,
    create_ik_controller,
    get_robot_configuration_snapshot,
    get_shared_configuration_snapshot,
)


DEFAULT_RANDOM_SEED = 20260805
DEFAULT_FK_BOUND_SAMPLE_COUNT = 20_000
DEFAULT_BOUND_PADDING_METERS = 0.02
DEFAULT_POSITION_SAMPLE_COUNT = 1_000
DEFAULT_ORIENTATION_SAMPLE_COUNT = 30
DEFAULT_STATIC_IK_SEED_COUNT = 5
DEFAULT_DEXTEROUS_COVERAGE_THRESHOLD = 0.95

FRONT_SECTOR_TOTAL_DEGREES = 180.0
MINIMUM_HEIGHT_ABOVE_BASE_METERS = 0.02
MINIMUM_SAMPLED_WORLD_Z = 0.07

CATEGORY_UNREACHABLE = 0
CATEGORY_PARTIAL = 1
CATEGORY_DEXTEROUS = 2

CATEGORY_NAMES = {
    CATEGORY_UNREACHABLE: "unreachable",
    CATEGORY_PARTIAL: "partially_reachable",
    CATEGORY_DEXTEROUS: "approximately_dexterous",
}

CATEGORY_COLORS = {
    CATEGORY_UNREACHABLE: (135, 135, 135),
    CATEGORY_PARTIAL: (245, 190, 30),
    CATEGORY_DEXTEROUS: (210, 50, 45),
}


@dataclass(frozen=True)
class SamplingConfig:
    random_seed: int
    fk_bound_sample_count: int
    bound_padding_meters: float
    position_sample_count: int
    orientation_sample_count: int
    static_ik_seed_count: int
    dexterous_coverage_threshold: float
    progress_every: int
    output_directory: Path
    create_html: bool


@dataclass
class StaticIKCandidate:
    accepted: bool
    arm_joint_angles: np.ndarray
    position_error: float
    orientation_error_degrees: float


def to_json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    return value


def generate_uniform_quaternions(
    random_generator: np.random.Generator,
    sample_count: int,
) -> np.ndarray:
    """Generate uniformly distributed unit quaternions in xyzw order."""

    samples = random_generator.random((sample_count, 3))
    first = samples[:, 0]
    second = samples[:, 1]
    third = samples[:, 2]

    quaternions = np.column_stack(
        [
            np.sqrt(1.0 - first) * np.sin(2.0 * math.pi * second),
            np.sqrt(1.0 - first) * np.cos(2.0 * math.pi * second),
            np.sqrt(first) * np.sin(2.0 * math.pi * third),
            np.sqrt(first) * np.cos(2.0 * math.pi * third),
        ]
    )

    return quaternions.astype(np.float64)


def set_arm_joint_state(
    controller: ConstrainedIKController,
    arm_joint_angles: np.ndarray,
) -> None:
    for joint_index, joint_angle in zip(
        controller.arm_joint_indices,
        arm_joint_angles,
    ):
        p.resetJointState(
            controller.body_id,
            int(joint_index),
            targetValue=float(joint_angle),
            targetVelocity=0.0,
        )


def read_link_pose(
    controller: ConstrainedIKController,
) -> Tuple[np.ndarray, np.ndarray]:
    state = p.getLinkState(
        controller.body_id,
        controller.ee_link,
        computeForwardKinematics=True,
    )
    return (
        np.asarray(state[4], dtype=np.float64),
        np.asarray(state[5], dtype=np.float64),
    )


def sample_fk_bounds(
    controller: ConstrainedIKController,
    random_generator: np.random.Generator,
    sample_count: int,
    padding: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_limits = controller.arm_soft_lower_limits
    upper_limits = controller.arm_soft_upper_limits
    original_arm_angles = controller.get_current_arm_joint_angles()

    joint_samples = random_generator.uniform(
        low=lower_limits,
        high=upper_limits,
        size=(sample_count, len(controller.arm_joint_indices)),
    )
    positions = np.empty((sample_count, 3), dtype=np.float64)

    try:
        for index, joint_angles in enumerate(joint_samples):
            set_arm_joint_state(controller, joint_angles)
            positions[index] = read_link_pose(controller)[0]
    finally:
        set_arm_joint_state(controller, original_arm_angles)

    minimum = np.min(positions, axis=0) - float(padding)
    maximum = np.max(positions, axis=0) + float(padding)
    return minimum, maximum, positions


def sample_front_workspace_positions(
    random_generator: np.random.Generator,
    sample_count: int,
    minimum: np.ndarray,
    maximum: np.ndarray,
    base_position: np.ndarray,
    forward_xy: np.ndarray,
) -> Tuple[np.ndarray, float]:
    minimum = np.asarray(minimum, dtype=np.float64).copy()
    maximum = np.asarray(maximum, dtype=np.float64).copy()
    base_position = np.asarray(base_position, dtype=np.float64)
    forward_xy = np.asarray(forward_xy, dtype=np.float64)

    minimum[2] = max(
        minimum[2],
        float(base_position[2]) + MINIMUM_HEIGHT_ABOVE_BASE_METERS,
        MINIMUM_SAMPLED_WORLD_Z,
    )
    if minimum[2] >= maximum[2]:
        raise RuntimeError("The sampled Z range is empty after height limits.")

    forward_norm = float(np.linalg.norm(forward_xy))
    if forward_norm < 1e-8:
        forward_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        forward_xy = forward_xy / forward_norm

    half_angle = math.radians(FRONT_SECTOR_TOTAL_DEGREES / 2.0)
    minimum_cosine = math.cos(half_angle)
    accepted_batches = []
    total_drawn = 0
    total_accepted = 0

    while total_accepted < sample_count:
        batch_size = max(1024, (sample_count - total_accepted) * 4)
        candidates = random_generator.uniform(
            low=minimum,
            high=maximum,
            size=(batch_size, 3),
        )
        relative_xy = candidates[:, :2] - base_position[:2]
        radii = np.linalg.norm(relative_xy, axis=1)
        nonzero = radii > 1e-8
        direction_cosines = np.full(batch_size, -1.0, dtype=np.float64)
        direction_cosines[nonzero] = (
            relative_xy[nonzero] @ forward_xy
        ) / radii[nonzero]
        mask = nonzero & (direction_cosines >= minimum_cosine)
        accepted = candidates[mask]
        accepted_batches.append(accepted)
        total_drawn += batch_size
        total_accepted += len(accepted)

    positions = np.concatenate(accepted_batches, axis=0)[:sample_count]
    return positions, float(total_accepted / total_drawn)


def generate_ik_seed_set(
    random_generator: np.random.Generator,
    neutral_arm_angles: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    seed_count: int,
) -> np.ndarray:
    seeds = [np.asarray(neutral_arm_angles, dtype=np.float64).copy()]
    if seed_count > 1:
        random_seeds = random_generator.uniform(
            low=lower_limits,
            high=upper_limits,
            size=(seed_count - 1, len(neutral_arm_angles)),
        )
        seeds.extend(random_seeds)
    return np.asarray(seeds, dtype=np.float64)


def calculate_static_candidate(
    controller: ConstrainedIKController,
    target_position: np.ndarray,
    arm_seed: np.ndarray,
    target_quaternion: Optional[np.ndarray] = None,
) -> StaticIKCandidate:
    set_arm_joint_state(controller, arm_seed)
    movable_angles = controller.get_current_movable_joint_angles()
    rest_poses = movable_angles.copy()
    rest_poses[controller.arm_dof_positions] = arm_seed

    arguments: Dict[str, Any] = {
        "bodyUniqueId": controller.body_id,
        "endEffectorLinkIndex": controller.ee_link,
        "targetPosition": np.asarray(target_position, dtype=np.float64).tolist(),
        "lowerLimits": controller.lower_limits.tolist(),
        "upperLimits": controller.upper_limits.tolist(),
        "jointRanges": controller.joint_ranges.tolist(),
        "restPoses": rest_poses.tolist(),
        "jointDamping": controller.joint_damping.tolist(),
        "solver": p.IK_DLS,
        "maxNumIterations": controller.maximum_iterations,
        "residualThreshold": controller.residual_threshold,
    }
    if target_quaternion is not None:
        arguments["targetOrientation"] = np.asarray(
            target_quaternion,
            dtype=np.float64,
        ).tolist()

    solution = np.asarray(
        p.calculateInverseKinematics(**arguments),
        dtype=np.float64,
    )
    if solution.size < len(controller.movable_joint_indices):
        return StaticIKCandidate(
            accepted=False,
            arm_joint_angles=np.asarray(arm_seed, dtype=np.float64).copy(),
            position_error=float("inf"),
            orientation_error_degrees=float("inf"),
        )

    candidate = solution[controller.arm_dof_positions]
    if not np.all(np.isfinite(candidate)):
        return StaticIKCandidate(
            accepted=False,
            arm_joint_angles=np.asarray(arm_seed, dtype=np.float64).copy(),
            position_error=float("inf"),
            orientation_error_degrees=float("inf"),
        )

    candidate = np.clip(
        candidate,
        controller.arm_soft_lower_limits,
        controller.arm_soft_upper_limits,
    )
    actual_position, actual_quaternion = controller._calculate_candidate_fk(
        candidate
    )
    position_error = float(
        np.linalg.norm(actual_position - np.asarray(target_position))
    )

    if target_quaternion is None:
        rotation_error = 0.0
    else:
        rotation_error = orientation_error_degrees(
            np.asarray(target_quaternion, dtype=np.float64),
            actual_quaternion,
        )

    accepted = position_error <= controller.maximum_position_error
    if target_quaternion is not None:
        accepted = accepted and (
            rotation_error <= controller.maximum_orientation_error_degrees
        )

    return StaticIKCandidate(
        accepted=bool(accepted),
        arm_joint_angles=candidate,
        position_error=position_error,
        orientation_error_degrees=float(rotation_error),
    )


def solve_with_seeds(
    controller: ConstrainedIKController,
    target_position: np.ndarray,
    arm_seeds: np.ndarray,
    target_quaternion: Optional[np.ndarray] = None,
) -> StaticIKCandidate:
    best: Optional[StaticIKCandidate] = None

    for seed in arm_seeds:
        candidate = calculate_static_candidate(
            controller=controller,
            target_position=target_position,
            target_quaternion=target_quaternion,
            arm_seed=seed,
        )
        if best is None:
            best = candidate
        elif target_quaternion is None:
            if candidate.position_error < best.position_error:
                best = candidate
        else:
            candidate_score = (
                candidate.position_error
                + math.radians(candidate.orientation_error_degrees) * 0.02
            )
            best_score = (
                best.position_error
                + math.radians(best.orientation_error_degrees) * 0.02
            )
            if candidate_score < best_score:
                best = candidate

        if candidate.accepted:
            return candidate

    if best is None:
        raise RuntimeError("No IK seeds were provided.")
    return best


def write_summary_csv(
    path: Path,
    positions: np.ndarray,
    position_reachable: np.ndarray,
    best_position_errors: np.ndarray,
    accepted_counts: np.ndarray,
    orientation_sample_count: int,
    acceptance_rates: np.ndarray,
    category_codes: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "position_index",
                "x",
                "y",
                "z",
                "position_reachable",
                "best_position_error",
                "orientation_sample_count",
                "accepted_orientation_count",
                "orientation_acceptance_rate",
                "category_code",
                "category_name",
            ]
        )
        for index, position in enumerate(positions):
            tested_count = orientation_sample_count if position_reachable[index] else 0
            writer.writerow(
                [
                    index,
                    *position.tolist(),
                    bool(position_reachable[index]),
                    float(best_position_errors[index]),
                    tested_count,
                    int(accepted_counts[index]),
                    float(acceptance_rates[index]),
                    int(category_codes[index]),
                    CATEGORY_NAMES[int(category_codes[index])],
                ]
            )


def write_ply(
    path: Path,
    positions: np.ndarray,
    category_codes: np.ndarray,
) -> None:
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("comment Panda-Allegro static IK workspace\n")
        file.write(f"element vertex {len(positions)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for position, code in zip(positions, category_codes):
            red, green, blue = CATEGORY_COLORS[int(code)]
            file.write(
                f"{position[0]:.9f} {position[1]:.9f} {position[2]:.9f} "
                f"{red} {green} {blue}\n"
            )


def write_html(
    path: Path,
    positions: np.ndarray,
    category_codes: np.ndarray,
    base_position: np.ndarray,
) -> None:
    traces = []
    for code in (
        CATEGORY_UNREACHABLE,
        CATEGORY_PARTIAL,
        CATEGORY_DEXTEROUS,
    ):
        selected = positions[category_codes == code]
        red, green, blue = CATEGORY_COLORS[code]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": CATEGORY_NAMES[code],
                "x": selected[:, 0].tolist(),
                "y": selected[:, 1].tolist(),
                "z": selected[:, 2].tolist(),
                "marker": {
                    "size": 3,
                    "opacity": 0.72,
                    "color": f"rgb({red},{green},{blue})",
                },
            }
        )
    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "robot_base",
            "x": [float(base_position[0])],
            "y": [float(base_position[1])],
            "z": [float(base_position[2])],
            "marker": {"size": 7, "color": "rgb(30,90,200)"},
        }
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Panda-Allegro theoretical workspace</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div id="workspace" style="width:100vw;height:100vh"></div>
  <script>
    const traces = {json.dumps(traces, separators=(',', ':'))};
    const layout = {{
      title: 'Panda-Allegro theoretical grasp TCP workspace',
      scene: {{
        aspectmode: 'data',
        xaxis: {{title: 'X (m)'}},
        yaxis: {{title: 'Y (m)'}},
        zaxis: {{title: 'Z (m)'}}
      }},
      margin: {{l:0,r:0,b:0,t:45}}
    }};
    Plotly.newPlot('workspace', traces, layout, {{responsive:true}});
  </script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def count_categories(category_codes: np.ndarray) -> Dict[str, int]:
    return {
        CATEGORY_NAMES[code]: int(np.sum(category_codes == code))
        for code in CATEGORY_NAMES
    }


def parse_arguments() -> SamplingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo sampling of the Panda-Allegro reachable and "
            "approximately dexterous static IK workspace."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--fk-bound-samples",
        type=int,
        default=DEFAULT_FK_BOUND_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--position-samples",
        type=int,
        default=DEFAULT_POSITION_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--orientation-samples",
        type=int,
        default=DEFAULT_ORIENTATION_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--ik-seeds",
        type=int,
        default=DEFAULT_STATIC_IK_SEED_COUNT,
    )
    parser.add_argument(
        "--dexterous-threshold",
        type=float,
        default=DEFAULT_DEXTEROUS_COVERAGE_THRESHOLD,
    )
    parser.add_argument(
        "--bound-padding",
        type=float,
        default=DEFAULT_BOUND_PADDING_METERS,
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ik_workspace",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not create the Plotly HTML preview.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small sample set for debugging the complete pipeline.",
    )
    arguments = parser.parse_args()

    if arguments.quick:
        arguments.fk_bound_samples = 2_000
        arguments.position_samples = 80
        arguments.orientation_samples = 8
        arguments.ik_seeds = 3
        arguments.progress_every = 10

    positive_values = {
        "fk_bound_samples": arguments.fk_bound_samples,
        "position_samples": arguments.position_samples,
        "orientation_samples": arguments.orientation_samples,
        "ik_seeds": arguments.ik_seeds,
        "progress_every": arguments.progress_every,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= arguments.dexterous_threshold <= 1.0:
        parser.error("--dexterous-threshold must be between 0 and 1")

    return SamplingConfig(
        random_seed=arguments.seed,
        fk_bound_sample_count=arguments.fk_bound_samples,
        bound_padding_meters=arguments.bound_padding,
        position_sample_count=arguments.position_samples,
        orientation_sample_count=arguments.orientation_samples,
        static_ik_seed_count=arguments.ik_seeds,
        dexterous_coverage_threshold=arguments.dexterous_threshold,
        progress_every=arguments.progress_every,
        output_directory=arguments.output_dir.resolve(),
        create_html=not arguments.no_html,
    )


def run_sampling(config: SamplingConfig) -> None:
    output_directory = config.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    samples_path = output_directory / "theoretical_workspace_samples.npz"
    summary_path = output_directory / "theoretical_workspace_summary.csv"
    metadata_path = output_directory / "theoretical_workspace_metadata.json"
    ply_path = output_directory / "theoretical_workspace_points.ply"
    html_path = output_directory / "panda_allegro_theoretical_workspace.html"

    random_generator = np.random.default_rng(config.random_seed)
    environment = create_environment(render_mode="rgb_array", renderer="Tiny")
    start_time = time.perf_counter()

    try:
        environment.reset()
        robot = environment.unwrapped.robot
        controller = create_ik_controller(robot)

        neutral_joint_values = np.asarray(
            robot.neutral_joint_values,
            dtype=np.float64,
        )
        neutral_arm_angles = np.asarray(
            robot.arm_neutral_joint_values,
            dtype=np.float64,
        )
        robot.set_joint_angles(neutral_joint_values)
        neutral_ee_position, neutral_ee_quaternion = read_link_pose(controller)
        base_position = np.asarray(
            p.getBasePositionAndOrientation(controller.body_id)[0],
            dtype=np.float64,
        )
        forward_xy = neutral_ee_position[:2] - base_position[:2]

        print("Sampling FK bounds...")
        sampling_minimum, sampling_maximum, fk_bound_positions = sample_fk_bounds(
            controller=controller,
            random_generator=random_generator,
            sample_count=config.fk_bound_sample_count,
            padding=config.bound_padding_meters,
        )
        robot.set_joint_angles(neutral_joint_values)

        positions, region_acceptance_ratio = sample_front_workspace_positions(
            random_generator=random_generator,
            sample_count=config.position_sample_count,
            minimum=sampling_minimum,
            maximum=sampling_maximum,
            base_position=base_position,
            forward_xy=forward_xy,
        )
        orientation_quaternions = generate_uniform_quaternions(
            random_generator,
            config.orientation_sample_count,
        )
        arm_seeds = generate_ik_seed_set(
            random_generator=random_generator,
            neutral_arm_angles=neutral_arm_angles,
            lower_limits=controller.arm_soft_lower_limits,
            upper_limits=controller.arm_soft_upper_limits,
            seed_count=config.static_ik_seed_count,
        )

        position_reachable = np.zeros(
            config.position_sample_count,
            dtype=bool,
        )
        best_position_errors = np.full(
            config.position_sample_count,
            np.inf,
            dtype=np.float64,
        )
        orientation_accepted = np.zeros(
            (
                config.position_sample_count,
                config.orientation_sample_count,
            ),
            dtype=bool,
        )
        orientation_position_errors = np.full(
            orientation_accepted.shape,
            np.inf,
            dtype=np.float64,
        )
        orientation_rotation_errors = np.full(
            orientation_accepted.shape,
            np.inf,
            dtype=np.float64,
        )

        print("Sampling static IK workspace...")
        for position_index, target_position in enumerate(positions):
            position_result = solve_with_seeds(
                controller=controller,
                target_position=target_position,
                arm_seeds=arm_seeds,
            )
            position_reachable[position_index] = position_result.accepted
            best_position_errors[position_index] = position_result.position_error

            if position_result.accepted:
                pose_seed_order = np.vstack(
                    [position_result.arm_joint_angles, arm_seeds]
                )[: config.static_ik_seed_count]
                for orientation_index, quaternion in enumerate(
                    orientation_quaternions
                ):
                    pose_result = solve_with_seeds(
                        controller=controller,
                        target_position=target_position,
                        target_quaternion=quaternion,
                        arm_seeds=pose_seed_order,
                    )
                    orientation_accepted[
                        position_index,
                        orientation_index,
                    ] = pose_result.accepted
                    orientation_position_errors[
                        position_index,
                        orientation_index,
                    ] = pose_result.position_error
                    orientation_rotation_errors[
                        position_index,
                        orientation_index,
                    ] = pose_result.orientation_error_degrees

            completed = position_index + 1
            if (
                completed % config.progress_every == 0
                or completed == config.position_sample_count
            ):
                print(
                    f"  completed {completed}/{config.position_sample_count} "
                    f"positions"
                )

        accepted_counts = np.sum(orientation_accepted, axis=1).astype(np.int32)
        acceptance_rates = accepted_counts / float(
            config.orientation_sample_count
        )
        category_codes = np.full(
            config.position_sample_count,
            CATEGORY_UNREACHABLE,
            dtype=np.int8,
        )
        category_codes[position_reachable] = CATEGORY_PARTIAL
        category_codes[
            position_reachable
            & (acceptance_rates >= config.dexterous_coverage_threshold)
        ] = CATEGORY_DEXTEROUS

        elapsed_seconds = time.perf_counter() - start_time
        category_counts = count_categories(category_codes)

        np.savez_compressed(
            samples_path,
            positions=positions,
            category_codes=category_codes,
            position_reachable=position_reachable,
            best_position_errors=best_position_errors,
            orientation_quaternions=orientation_quaternions,
            orientation_accepted=orientation_accepted,
            orientation_position_errors=orientation_position_errors,
            orientation_rotation_errors_degrees=orientation_rotation_errors,
            orientation_acceptance_rates=acceptance_rates,
            fk_bound_positions=fk_bound_positions,
            sampling_minimum=sampling_minimum,
            sampling_maximum=sampling_maximum,
        )
        write_summary_csv(
            path=summary_path,
            positions=positions,
            position_reachable=position_reachable,
            best_position_errors=best_position_errors,
            accepted_counts=accepted_counts,
            orientation_sample_count=config.orientation_sample_count,
            acceptance_rates=acceptance_rates,
            category_codes=category_codes,
        )
        write_ply(ply_path, positions, category_codes)
        if config.create_html:
            write_html(html_path, positions, category_codes, base_position)

        metadata = {
            "analysis_type": "theoretical_static_grasp_tcp_workspace",
            "random_seed": config.random_seed,
            "fk_bound_sample_count": config.fk_bound_sample_count,
            "bound_padding_meters": config.bound_padding_meters,
            "position_sample_count": config.position_sample_count,
            "orientation_samples_per_reachable_position": (
                config.orientation_sample_count
            ),
            "static_ik_seed_count": config.static_ik_seed_count,
            "dexterous_coverage_threshold": (
                config.dexterous_coverage_threshold
            ),
            "front_sector_total_degrees": FRONT_SECTOR_TOTAL_DEGREES,
            "minimum_height_above_base_meters": (
                MINIMUM_HEIGHT_ABOVE_BASE_METERS
            ),
            "minimum_sampled_world_z": MINIMUM_SAMPLED_WORLD_Z,
            "sampling_minimum": sampling_minimum,
            "sampling_maximum": sampling_maximum,
            "base_position": base_position,
            "neutral_ee_position": neutral_ee_position,
            "neutral_ee_quaternion_xyzw": neutral_ee_quaternion,
            "detected_forward_xy": forward_xy,
            "region_sampling_acceptance_ratio": region_acceptance_ratio,
            "category_names": CATEGORY_NAMES,
            "category_counts": category_counts,
            "workspace_point_definition": "world position of grasp_tcp",
            "excluded_dynamic_ik_constraints": [
                "maximum_candidate_joint_jump",
                "maximum_joint_step",
                "maximum_joint_velocity",
                "maximum_joint_acceleration",
                "continuity_history",
            ],
            "pose_filter_used_for_classification": False,
            "collision_checking_used": False,
            "elapsed_seconds": elapsed_seconds,
            "shared_configuration": get_shared_configuration_snapshot(),
            "robot_configuration": get_robot_configuration_snapshot(robot),
        }
        metadata_path.write_text(
            json.dumps(
                to_json_compatible(metadata),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("-" * 72)
        print(
            "Reachable positions:",
            int(np.sum(position_reachable)),
            "/",
            config.position_sample_count,
        )
        for category_name, count in category_counts.items():
            print(f"{category_name}: {count}")
        print(f"Elapsed seconds: {elapsed_seconds:.2f}")
        print(f"Saved: {samples_path}")
        print(f"Saved: {summary_path}")
        print(f"Saved: {metadata_path}")
        print(f"Saved: {ply_path}")
        if config.create_html:
            print(f"Saved: {html_path}")
    finally:
        environment.close()


def main() -> None:
    config = parse_arguments()
    print("=" * 72)
    print("Panda-Allegro Monte Carlo static IK workspace sampling")
    print("=" * 72)
    print(to_json_compatible(config.__dict__))
    run_sampling(config)


if __name__ == "__main__":
    main()
