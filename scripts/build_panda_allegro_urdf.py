from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_ASSET_DIR = (
    PROJECT_ROOT
    / "vendor"
    / "panda-gym"
    / "panda_gym"
    / "assets"
    / "robots"
    / "panda_allegro"
)

PANDA_URDF = (
    ROBOT_ASSET_DIR
    / "franka_panda"
    / "panda.urdf"
)

ALLEGRO_URDF = (
    ROBOT_ASSET_DIR
    / "allegro_hand"
    / "allegro_hand_right.urdf"
)

OUTPUT_URDF = ROBOT_ASSET_DIR / "panda_allegro.urdf"


# ============================================================
# Allegro Hand 安装参数
# ============================================================

BASE_MOUNT_XYZ = np.array(
    [0.0, 0.0, 0.095],
    dtype=np.float64,
)

# 已经确认握拳方向正确的基础安装姿态
BASE_MOUNT_RPY = np.array(
    [
        math.pi,
        math.pi / 2.0,
        -math.pi / 4.0,
    ],
    dtype=np.float64,
)

# 让手掌相对原构型立起 90°
STANDUP_LOCAL_RPY = np.array(
    [
        0.0,
        -math.pi / 2.0,
        0.0,
    ],
    dtype=np.float64,
)

MOUNT_XYZ = BASE_MOUNT_XYZ.copy()


# ============================================================
# 抓取中心参数
# ============================================================

# 抓取中心相对于 palm 坐标系的初始偏移。
# 后续只微调 GRASP_TCP_XYZ。
GRASP_TCP_PARENT_LINK = "palm"

GRASP_TCP_XYZ = np.array(
    [0.0, 0.0, 0.075],
    dtype=np.float64,
)

GRASP_TCP_RPY = np.array(
    [0.0, 0.0, 0.0],
    dtype=np.float64,
)


# ============================================================
# 旋转转换
# ============================================================

def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )

    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )

    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rotation_z @ rotation_y @ rotation_x


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = math.sqrt(
        rotation[0, 0] ** 2
        + rotation[1, 0] ** 2
    )

    singular = sy < 1e-8

    if not singular:
        roll = math.atan2(
            rotation[2, 1],
            rotation[2, 2],
        )

        pitch = math.atan2(
            -rotation[2, 0],
            sy,
        )

        yaw = math.atan2(
            rotation[1, 0],
            rotation[0, 0],
        )

    else:
        roll = math.atan2(
            -rotation[1, 2],
            rotation[1, 1],
        )

        pitch = math.atan2(
            -rotation[2, 0],
            sy,
        )

        yaw = 0.0

    return np.array(
        [roll, pitch, yaw],
        dtype=np.float64,
    )


def calculate_mount_rpy() -> np.ndarray:
    base_rotation = rpy_to_matrix(BASE_MOUNT_RPY)
    standup_rotation = rpy_to_matrix(STANDUP_LOCAL_RPY)

    final_rotation = base_rotation @ standup_rotation

    return matrix_to_rpy(final_rotation)


MOUNT_RPY = calculate_mount_rpy()


# ============================================================
# 通用函数
# ============================================================

def vector_to_text(vector: np.ndarray) -> str:
    return " ".join(
        f"{float(value):.8f}"
        for value in vector
    )


def _suffix_match_score(
    candidate: Path,
    source_parts: Tuple[str, ...],
) -> int:
    candidate_parts = tuple(
        part.lower()
        for part in candidate.parts
    )

    normalized_source_parts = tuple(
        part.lower()
        for part in source_parts
    )

    score = 0

    for candidate_part, source_part in zip(
        reversed(candidate_parts),
        reversed(normalized_source_parts),
    ):
        if candidate_part != source_part:
            break

        score += 1

    return score


def rewrite_mesh_paths(
    root: ET.Element,
    source_urdf: Path,
    output_urdf: Path,
) -> None:
    source_dir = source_urdf.parent.resolve()
    output_dir = output_urdf.parent.resolve()

    mesh_count = 0

    for mesh in root.findall(".//mesh"):
        raw_filename = mesh.attrib.get("filename")

        if not raw_filename:
            continue

        mesh_count += 1

        normalized = raw_filename.replace(
            "\\",
            "/",
        ).strip()

        if normalized.startswith("package://"):
            path_text = normalized[len("package://"):]

        elif normalized.startswith("model://"):
            path_text = normalized[len("model://"):]

        elif normalized.startswith("file://"):
            path_text = normalized[len("file://"):]

        else:
            path_text = normalized

        source_parts = PurePosixPath(path_text).parts
        filename = PurePosixPath(path_text).name

        candidate_paths: List[Path] = []

        raw_path = Path(path_text)

        if raw_path.is_absolute():
            candidate_paths.append(raw_path)

        else:
            candidate_paths.append(
                source_dir.joinpath(*source_parts)
            )

            if len(source_parts) > 1:
                candidate_paths.append(
                    source_dir.joinpath(
                        *source_parts[1:]
                    )
                )

            candidate_paths.append(
                source_dir
                / "meshes"
                / Path(*source_parts)
            )

            if len(source_parts) > 1:
                candidate_paths.append(
                    source_dir
                    / "meshes"
                    / Path(*source_parts[1:])
                )

        resolved_path: Optional[Path] = None

        for candidate in candidate_paths:
            candidate = candidate.resolve()

            if candidate.exists() and candidate.is_file():
                resolved_path = candidate
                break

        if resolved_path is None:
            recursive_matches = [
                path.resolve()
                for path in source_dir.rglob(filename)
                if path.is_file()
            ]

            if recursive_matches:
                recursive_matches.sort(
                    key=lambda path: _suffix_match_score(
                        path,
                        source_parts,
                    ),
                    reverse=True,
                )

                resolved_path = recursive_matches[0]

        if resolved_path is None:
            raise FileNotFoundError(
                "\nCannot resolve mesh file.\n"
                f"Source URDF: {source_urdf}\n"
                f"Mesh entry: {raw_filename}\n"
                f"Search root: {source_dir}"
            )

        relative_path = os.path.relpath(
            resolved_path,
            output_dir,
        ).replace("\\", "/")

        mesh.set(
            "filename",
            relative_path,
        )

        print(
            f"Mesh fixed: {raw_filename} "
            f"-> {relative_path}"
        )

    print(
        f"Resolved {mesh_count} mesh entries "
        f"from {source_urdf.name}."
    )


def validate_output_mesh_paths(
    output_urdf: Path,
) -> None:
    tree = ET.parse(output_urdf)
    root = tree.getroot()

    output_dir = output_urdf.parent.resolve()

    total = 0
    missing: List[Tuple[str, Path]] = []

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")

        if not filename:
            continue

        total += 1

        normalized = filename.replace("\\", "/")

        mesh_path = (
            output_dir / normalized
        ).resolve()

        if not mesh_path.exists():
            missing.append(
                (filename, mesh_path)
            )

    print()
    print("=" * 80)
    print("Mesh path validation")
    print("=" * 80)
    print(f"Total mesh entries: {total}")

    print(
        f"Existing mesh files: "
        f"{total - len(missing)}"
    )

    print(
        f"Missing mesh files: "
        f"{len(missing)}"
    )

    if missing:
        for filename, mesh_path in missing:
            print(f"MISSING: {filename}")
            print(f"         {mesh_path}")

        raise FileNotFoundError(
            "Generated URDF contains missing mesh files."
        )

    print("Mesh validation passed.")


def remove_element_by_name(
    root: ET.Element,
    tag: str,
    names: set,
) -> None:
    for element in list(root):
        if (
            element.tag == tag
            and element.attrib.get("name") in names
        ):
            print(
                f"Removed Panda {tag}: "
                f"{element.attrib.get('name')}"
            )

            root.remove(element)


def element_name_exists(
    root: ET.Element,
    tag: str,
    name: str,
) -> bool:
    return any(
        element.attrib.get("name") == name
        for element in root.findall(tag)
    )


# ============================================================
# 添加 Panda 与 Allegro 的连接关节
# ============================================================

def add_mount_joint(
    root: ET.Element,
) -> None:
    joint = ET.Element(
        "joint",
        {
            "name": "panda_allegro_mount_joint",
            "type": "fixed",
        },
    )

    ET.SubElement(
        joint,
        "parent",
        {
            "link": "panda_link8",
        },
    )

    ET.SubElement(
        joint,
        "child",
        {
            "link": "base_link",
        },
    )

    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": vector_to_text(MOUNT_XYZ),
            "rpy": vector_to_text(MOUNT_RPY),
        },
    )

    root.append(joint)


# ============================================================
# 添加腕部 TCP
# ============================================================

def add_allegro_tcp(
    root: ET.Element,
) -> None:
    if element_name_exists(
        root,
        "link",
        "allegro_tcp",
    ):
        print("allegro_tcp already exists.")
        return

    tcp_link = ET.Element(
        "link",
        {
            "name": "allegro_tcp",
        },
    )

    root.append(tcp_link)

    tcp_joint = ET.Element(
        "joint",
        {
            "name": "allegro_tcp_joint",
            "type": "fixed",
        },
    )

    ET.SubElement(
        tcp_joint,
        "parent",
        {
            "link": "wrist",
        },
    )

    ET.SubElement(
        tcp_joint,
        "child",
        {
            "link": "allegro_tcp",
        },
    )

    ET.SubElement(
        tcp_joint,
        "origin",
        {
            "xyz": "0 0 0",
            "rpy": "0 0 0",
        },
    )

    root.append(tcp_joint)


# ============================================================
# 添加抓取中心 TCP
# ============================================================

def add_grasp_tcp(
    root: ET.Element,
) -> None:
    if element_name_exists(
        root,
        "link",
        "grasp_tcp",
    ):
        print("grasp_tcp already exists.")
        return

    if not element_name_exists(
        root,
        "link",
        GRASP_TCP_PARENT_LINK,
    ):
        raise RuntimeError(
            f"Cannot find grasp TCP parent link: "
            f"{GRASP_TCP_PARENT_LINK}"
        )

    grasp_link = ET.Element(
        "link",
        {
            "name": "grasp_tcp",
        },
    )

    root.append(grasp_link)

    grasp_joint = ET.Element(
        "joint",
        {
            "name": "grasp_tcp_joint",
            "type": "fixed",
        },
    )

    ET.SubElement(
        grasp_joint,
        "parent",
        {
            "link": GRASP_TCP_PARENT_LINK,
        },
    )

    ET.SubElement(
        grasp_joint,
        "child",
        {
            "link": "grasp_tcp",
        },
    )

    ET.SubElement(
        grasp_joint,
        "origin",
        {
            "xyz": vector_to_text(GRASP_TCP_XYZ),
            "rpy": vector_to_text(GRASP_TCP_RPY),
        },
    )

    root.append(grasp_joint)


# ============================================================
# 构建 Panda + 直立 Allegro
# ============================================================

def build_panda_allegro_urdf() -> None:
    print("=" * 80)
    print("Building upright Panda + Allegro URDF")
    print("=" * 80)

    print("Panda URDF:", PANDA_URDF)
    print("Allegro URDF:", ALLEGRO_URDF)
    print("Output URDF:", OUTPUT_URDF)

    if not PANDA_URDF.exists():
        raise FileNotFoundError(
            f"Panda URDF does not exist: {PANDA_URDF}"
        )

    if not ALLEGRO_URDF.exists():
        raise FileNotFoundError(
            f"Allegro URDF does not exist: {ALLEGRO_URDF}"
        )

    panda_tree = ET.parse(PANDA_URDF)
    panda_root = panda_tree.getroot()

    allegro_tree = ET.parse(ALLEGRO_URDF)
    allegro_root = allegro_tree.getroot()

    rewrite_mesh_paths(
        root=panda_root,
        source_urdf=PANDA_URDF,
        output_urdf=OUTPUT_URDF,
    )

    rewrite_mesh_paths(
        root=allegro_root,
        source_urdf=ALLEGRO_URDF,
        output_urdf=OUTPUT_URDF,
    )

    panda_links_to_remove = {
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
        "panda_grasptarget",
    }

    panda_joints_to_remove = {
        "panda_hand_joint",
        "panda_finger_joint1",
        "panda_finger_joint2",
        "panda_grasptarget_hand",
    }

    remove_element_by_name(
        root=panda_root,
        tag="link",
        names=panda_links_to_remove,
    )

    remove_element_by_name(
        root=panda_root,
        tag="joint",
        names=panda_joints_to_remove,
    )

    for child in list(allegro_root):
        panda_root.append(
            copy.deepcopy(child)
        )

    add_mount_joint(panda_root)
    add_allegro_tcp(panda_root)
    add_grasp_tcp(panda_root)

    OUTPUT_URDF.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(ET, "indent"):
        ET.indent(
            panda_tree,
            space="  ",
        )

    panda_tree.write(
        OUTPUT_URDF,
        encoding="utf-8",
        xml_declaration=True,
    )

    validate_output_mesh_paths(
        OUTPUT_URDF
    )

    print()
    print("=" * 80)
    print("Build completed")
    print("=" * 80)

    print(
        "Base MOUNT_XYZ:",
        vector_to_text(BASE_MOUNT_XYZ),
    )

    print(
        "Base MOUNT_RPY:",
        vector_to_text(BASE_MOUNT_RPY),
    )

    print(
        "Stand-up local RPY:",
        vector_to_text(STANDUP_LOCAL_RPY),
    )

    print(
        "Final MOUNT_XYZ:",
        vector_to_text(MOUNT_XYZ),
    )

    print(
        "Final MOUNT_RPY:",
        vector_to_text(MOUNT_RPY),
    )

    print(
        "Grasp TCP parent:",
        GRASP_TCP_PARENT_LINK,
    )

    print(
        "Grasp TCP XYZ:",
        vector_to_text(GRASP_TCP_XYZ),
    )

    print(
        "Grasp TCP RPY:",
        vector_to_text(GRASP_TCP_RPY),
    )

    print("Generated:", OUTPUT_URDF)


if __name__ == "__main__":
    build_panda_allegro_urdf()
