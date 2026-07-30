import time

import cv2
import gymnasium as gym
import mediapipe as mp
import numpy as np
import pybullet as p

import panda_gym


CAMERA_ID = 0
MIRROR_CAMERA = True

# ============================================================
# Translation mapping
# ============================================================

# camera x/y -> robot y/z
GAIN_IMAGE_X_TO_ROBOT_Y = 1.0
GAIN_IMAGE_Y_TO_ROBOT_Z = 1.0

# hand scale -> robot x, 用于前后移动
GAIN_DEPTH_TO_ROBOT_X = 0.55

# workspace limit relative to calibrated robot TCP pose
MAX_DELTA_X = 0.30
MAX_DELTA_Y = 0.30
MAX_DELTA_Z = 0.30

# dead zone，减少手部微抖
DEADZONE_IMAGE = 0.015
DEADZONE_DEPTH = 0.035

# 位置低通滤波，越大越跟手，越小越稳
POS_FILTER_ALPHA = 0.35

# ============================================================
# Rotation mapping
# ============================================================

USE_WRIST_ROTATION = True

# 旋转放大/缩小系数
ROTATION_GAIN = 0.85

# 最大允许 wrist rotation，防止 IK 突然翻转
MAX_ROTATION_DEG = 75.0

# 姿态低通滤波，越大越跟手，越小越稳
ROT_FILTER_ALPHA = 0.35

# ============================================================
# Hand control
# ============================================================

MAX_HAND_ACTION = 1.0

ALLEGRO_LOWER = np.array(
    [
        -0.470, -0.196, -0.174, -0.227,
        -0.470, -0.196, -0.174, -0.227,
        -0.470, -0.196, -0.174, -0.227,
         0.263, -0.105, -0.189, -0.162,
    ],
    dtype=np.float32,
)

ALLEGRO_UPPER = np.array(
    [
        0.470, 1.610, 1.709, 1.618,
        0.470, 1.610, 1.709, 1.618,
        0.470, 1.610, 1.709, 1.618,
        1.396, 1.163, 1.644, 1.719,
    ],
    dtype=np.float32,
)


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def apply_deadzone(x, dz):
    if abs(x) < dz:
        return 0.0
    return x


def lm_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def lm_to_pixel(lm, w, h):
    return np.array([int(lm.x * w), int(lm.y * h)], dtype=np.int32)


def safe_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    cos_value = np.dot(v1, v2) / (n1 * n2)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return np.degrees(np.arccos(cos_value))


# ============================================================
# Quaternion / rotation matrix utils
# PyBullet quaternion format: [x, y, z, w]
# ============================================================

def quat_to_matrix(q):
    return np.array(p.getMatrixFromQuaternion(q), dtype=np.float32).reshape(3, 3)


def matrix_to_quat(R):
    R = np.asarray(R, dtype=np.float64)

    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    return q


def rotation_matrix_to_axis_angle(R):
    R = np.asarray(R, dtype=np.float32)

    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = float(np.arccos(cos_angle))

    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float32,
    )

    axis = axis / (2.0 * np.sin(angle) + 1e-8)
    axis = normalize(axis)

    return axis, angle


def axis_angle_to_rotation_matrix(axis, angle):
    axis = normalize(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )

    return R


def limit_and_scale_rotation(R_rel):
    axis, angle = rotation_matrix_to_axis_angle(R_rel)

    max_angle = np.deg2rad(MAX_ROTATION_DEG)
    angle = np.clip(angle * ROTATION_GAIN, -max_angle, max_angle)

    return axis_angle_to_rotation_matrix(axis, angle)


def slerp_quat(q1, q2, alpha):
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)

    dot = float(np.dot(q1, q2))

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.995:
        q = q1 + alpha * (q2 - q1)
        return q / (np.linalg.norm(q) + 1e-8)

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = s0 * q1 + s1 * q2
    return q / (np.linalg.norm(q) + 1e-8)


# ============================================================
# Hand pose estimation
# ============================================================

def estimate_palm_frame_3d(landmarks):
    """
    Estimate a local 3D palm/wrist frame from MediaPipe landmarks.

    Frame definition:
        X axis: pinky_mcp -> index_mcp
        Y axis: wrist -> middle_mcp
        Z axis: palm normal = X cross Y

    This is still an approximate visual hand frame,
    but much better than the previous pure 2D wrist-axis drawing.
    """
    wrist = lm_to_np(landmarks[0])
    index_mcp = lm_to_np(landmarks[5])
    middle_mcp = lm_to_np(landmarks[9])
    pinky_mcp = lm_to_np(landmarks[17])

    x_axis = normalize(index_mcp - pinky_mcp)
    y_axis_raw = normalize(middle_mcp - wrist)

    z_axis = normalize(np.cross(x_axis, y_axis_raw))
    y_axis = normalize(np.cross(z_axis, x_axis))

    R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)

    return R


def get_palm_center(landmarks):
    wrist = lm_to_np(landmarks[0])
    index_mcp = lm_to_np(landmarks[5])
    middle_mcp = lm_to_np(landmarks[9])
    ring_mcp = lm_to_np(landmarks[13])
    pinky_mcp = lm_to_np(landmarks[17])

    return (wrist + index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 5.0


def get_hand_scale(landmarks):
    """
    Estimate front/back movement from hand scale.

    Larger palm width usually means hand is closer to camera.
    This is more reliable than directly using MediaPipe z alone.
    """
    index_mcp = lm_to_np(landmarks[5])
    pinky_mcp = lm_to_np(landmarks[17])
    middle_mcp = lm_to_np(landmarks[9])
    wrist = lm_to_np(landmarks[0])

    palm_width = np.linalg.norm(index_mcp - pinky_mcp)
    palm_length = np.linalg.norm(middle_mcp - wrist)

    return float(0.6 * palm_width + 0.4 * palm_length)


def estimate_finger_curl(landmarks, ids):
    """
    Estimate one finger curl.

    ids = [mcp, pip, dip, tip]
    output:
        0 = open
        1 = curled
    """
    mcp = lm_to_np(landmarks[ids[0]])
    pip = lm_to_np(landmarks[ids[1]])
    dip = lm_to_np(landmarks[ids[2]])
    tip = lm_to_np(landmarks[ids[3]])

    v1 = mcp - pip
    v2 = tip - pip

    angle = safe_angle(v1, v2)

    curl_angle = (165.0 - angle) / (165.0 - 70.0)
    curl_angle = np.clip(curl_angle, 0.0, 1.0)

    wrist = lm_to_np(landmarks[0])
    dist_tip = np.linalg.norm(tip - wrist)
    dist_mcp = np.linalg.norm(mcp - wrist) + 1e-6
    ratio = dist_tip / dist_mcp

    curl_dist = 1.0 - np.clip((ratio - 1.1) / (2.2 - 1.1), 0.0, 1.0)

    curl = 0.75 * curl_angle + 0.25 * curl_dist
    return float(np.clip(curl, 0.0, 1.0))


def estimate_thumb_curl(landmarks):
    return estimate_finger_curl(landmarks, [1, 2, 3, 4])


def estimate_all_finger_curls(landmarks):
    index_curl = estimate_finger_curl(landmarks, [5, 6, 7, 8])
    middle_curl = estimate_finger_curl(landmarks, [9, 10, 11, 12])
    ring_curl = estimate_finger_curl(landmarks, [13, 14, 15, 16])
    pinky_curl = estimate_finger_curl(landmarks, [17, 18, 19, 20])
    thumb_curl = estimate_thumb_curl(landmarks)

    # Allegro 只有四根手指，这里让第三根参考 ring + pinky
    ring_curl = 0.7 * ring_curl + 0.3 * pinky_curl

    return np.array(
        [index_curl, middle_curl, ring_curl, thumb_curl],
        dtype=np.float32,
    )


def curls_to_allegro_target_q(curls):
    open_q = np.clip(np.zeros(16, dtype=np.float32), ALLEGRO_LOWER, ALLEGRO_UPPER)

    normal_weights = np.array([0.12, 0.80, 0.85, 0.85], dtype=np.float32)
    thumb_weights = np.array([0.35, 0.65, 0.75, 0.80], dtype=np.float32)

    target_q = open_q.copy()

    for finger_id in range(3):
        start = finger_id * 4
        end = start + 4

        curl = curls[finger_id]
        target_q[start:end] = open_q[start:end] + curl * normal_weights * (
            ALLEGRO_UPPER[start:end] - open_q[start:end]
        )

    start = 12
    end = 16

    curl = curls[3]
    target_q[start:end] = open_q[start:end] + curl * thumb_weights * (
        ALLEGRO_UPPER[start:end] - open_q[start:end]
    )

    return np.clip(target_q, ALLEGRO_LOWER, ALLEGRO_UPPER).astype(np.float32)


# ============================================================
# Robot direct IK control
# ============================================================

def get_robot_body_id(robot):
    return robot.sim._bodies_idx[robot.body_name]


def get_ee_pose(robot):
    body_id = get_robot_body_id(robot)
    state = p.getLinkState(body_id, robot.ee_link)

    # worldLinkFramePosition / worldLinkFrameOrientation
    if len(state) >= 6:
        pos = np.array(state[4], dtype=np.float32)
        quat = np.array(state[5], dtype=np.float32)
    else:
        pos = np.array(state[0], dtype=np.float32)
        quat = np.array(state[1], dtype=np.float32)

    return pos, quat


def control_robot_pose_and_hand(robot, target_pos, target_quat, target_hand_q):
    """
    Directly control Panda arm by IK with target position + orientation,
    then control Allegro joints by absolute q target.

    这里不用 env.step(action)，因为原 19 维 action 没有 wrist rotation。
    """
    target_arm_q = robot.inverse_kinematics(
        link=robot.ee_link,
        position=np.asarray(target_pos, dtype=np.float32),
        orientation=np.asarray(target_quat, dtype=np.float32),
    )

    target_arm_q = np.asarray(target_arm_q[:7], dtype=np.float32)

    target_angles = np.concatenate(
        [
            target_arm_q,
            np.asarray(target_hand_q, dtype=np.float32),
        ]
    )

    robot.control_joints(target_angles=target_angles)


# ============================================================
# Visualization
# ============================================================

def draw_wrist_axes(frame, landmarks, R_hand):
    h, w = frame.shape[:2]

    wrist_px = lm_to_pixel(landmarks[0], w, h)
    origin = wrist_px.astype(np.float32)

    axis_len = 65

    # R_hand columns: x, y, z
    x_dir = np.array([R_hand[0, 0], R_hand[1, 0]], dtype=np.float32)
    y_dir = np.array([R_hand[0, 1], R_hand[1, 1]], dtype=np.float32)
    z_dir = np.array([R_hand[0, 2], R_hand[1, 2]], dtype=np.float32)

    x_dir = normalize(x_dir)
    y_dir = normalize(y_dir)
    z_dir = normalize(z_dir)

    x_end = (origin + axis_len * x_dir).astype(int)
    y_end = (origin + axis_len * y_dir).astype(int)
    z_end = (origin + axis_len * z_dir).astype(int)

    origin_i = origin.astype(int)

    cv2.circle(frame, tuple(origin_i), 7, (255, 255, 255), -1)

    # OpenCV BGR:
    # X red, Y green, Z blue
    cv2.arrowedLine(frame, tuple(origin_i), tuple(x_end), (0, 0, 255), 3, tipLength=0.25)
    cv2.arrowedLine(frame, tuple(origin_i), tuple(y_end), (0, 255, 0), 3, tipLength=0.25)
    cv2.arrowedLine(frame, tuple(origin_i), tuple(z_end), (255, 0, 0), 3, tipLength=0.25)

    cv2.putText(frame, "X", tuple(x_end + np.array([5, 0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Y", tuple(y_end + np.array([5, 0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Z", tuple(z_end + np.array([5, 0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(
        frame,
        "wrist / palm frame",
        tuple(origin_i + np.array([10, -15])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def draw_status(frame, target_pos, curls, use_rotation):
    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        f"target pos: {np.round(target_pos, 2)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"curl I/M/R/T: {np.round(curls, 2)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"rotation IK: {use_rotation}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "C: calibrate | R: reset | T: toggle rotation | Q: quit",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def main():
    print("=" * 90)
    print("Camera -> PandaAllegro with 3D wrist/palm pose estimation")
    print("=" * 90)
    print("C : recalibrate current hand pose")
    print("R : reset environment")
    print("T : toggle wrist rotation IK on/off")
    print("Q : quit")
    print("=" * 90)

    env = gym.make(
        "PandaAllegroPickAndPlace-v0",
        render_mode="human",
    )

    observation, info = env.reset()
    robot = env.unwrapped.robot

    print("env.action_space:", env.action_space)
    print("robot.ee_link:", robot.ee_link)

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        env.close()
        raise RuntimeError(f"Cannot open camera {CAMERA_ID}. Try CAMERA_ID = 1 or 2.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    ref_palm = None
    ref_scale = None
    ref_hand_R = None
    ref_robot_pos = None
    ref_robot_quat = None
    ref_robot_R = None

    filtered_pos = None
    filtered_quat = None

    use_rotation = USE_WRIST_ROTATION
    last_print_time = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read camera frame.")
                continue

            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            target_pos = None
            target_quat = None
            curls = np.zeros(4, dtype=np.float32)
            target_hand_q = robot.get_allegro_joint_positions()

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                palm = get_palm_center(landmarks)
                scale = get_hand_scale(landmarks)
                hand_R = estimate_palm_frame_3d(landmarks)

                draw_wrist_axes(frame, landmarks, hand_R)

                if ref_palm is None:
                    ref_palm = palm.copy()
                    ref_scale = scale
                    ref_hand_R = hand_R.copy()

                    ref_robot_pos, ref_robot_quat = get_ee_pose(robot)
                    ref_robot_R = quat_to_matrix(ref_robot_quat)

                    filtered_pos = ref_robot_pos.copy()
                    filtered_quat = ref_robot_quat.copy()

                    print("Calibrated:")
                    print("  ref_palm:", ref_palm)
                    print("  ref_scale:", ref_scale)
                    print("  ref_robot_pos:", ref_robot_pos)
                    print("  ref_robot_quat:", ref_robot_quat)

                # ----------------------------
                # Translation
                # ----------------------------
                delta_palm = palm - ref_palm
                delta_scale = (scale - ref_scale) / (ref_scale + 1e-6)

                dx_depth = apply_deadzone(delta_scale, DEADZONE_DEPTH)
                dy_image = apply_deadzone(delta_palm[0], DEADZONE_IMAGE)
                dz_image = apply_deadzone(delta_palm[1], DEADZONE_IMAGE)

                # Mapping:
                # hand closer/farther -> robot x
                # image left/right    -> robot y
                # image up/down       -> robot z
                delta_robot = np.array(
                    [
                        dx_depth * GAIN_DEPTH_TO_ROBOT_X,
                        -dy_image * GAIN_IMAGE_X_TO_ROBOT_Y,
                        -dz_image * GAIN_IMAGE_Y_TO_ROBOT_Z,
                    ],
                    dtype=np.float32,
                )

                delta_robot[0] = np.clip(delta_robot[0], -MAX_DELTA_X, MAX_DELTA_X)
                delta_robot[1] = np.clip(delta_robot[1], -MAX_DELTA_Y, MAX_DELTA_Y)
                delta_robot[2] = np.clip(delta_robot[2], -MAX_DELTA_Z, MAX_DELTA_Z)

                raw_target_pos = ref_robot_pos + delta_robot

                if filtered_pos is None:
                    filtered_pos = raw_target_pos.copy()
                else:
                    filtered_pos = (
                        POS_FILTER_ALPHA * raw_target_pos
                        + (1.0 - POS_FILTER_ALPHA) * filtered_pos
                    )

                target_pos = filtered_pos.copy()

                # ----------------------------
                # Rotation
                # ----------------------------
                if use_rotation:
                    R_rel_hand = hand_R @ ref_hand_R.T
                    R_rel_hand = limit_and_scale_rotation(R_rel_hand)

                    raw_target_R = ref_robot_R @ R_rel_hand
                    raw_target_quat = matrix_to_quat(raw_target_R)

                    if filtered_quat is None:
                        filtered_quat = raw_target_quat.copy()
                    else:
                        filtered_quat = slerp_quat(
                            filtered_quat,
                            raw_target_quat,
                            ROT_FILTER_ALPHA,
                        )

                    target_quat = filtered_quat.copy()

                else:
                    target_quat = ref_robot_quat.copy()

                # ----------------------------
                # Finger curl -> Allegro
                # ----------------------------
                curls = estimate_all_finger_curls(landmarks)
                target_hand_q = curls_to_allegro_target_q(curls)

                control_robot_pose_and_hand(
                    robot=robot,
                    target_pos=target_pos,
                    target_quat=target_quat,
                    target_hand_q=target_hand_q,
                )

                now = time.time()
                if now - last_print_time > 0.4:
                    last_print_time = now
                    print(
                        "target_pos=",
                        np.round(target_pos, 3),
                        "depth_delta=",
                        round(float(delta_scale), 3),
                        "curls=",
                        np.round(curls, 3),
                        "rotation=",
                        use_rotation,
                    )

                draw_status(frame, target_pos, curls, use_rotation)

            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            robot.sim.step()

            cv2.imshow("Camera -> PandaAllegro 3D Wrist Pose", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                print("Quit.")
                break

            if key == ord("c") or key == ord("C"):
                ref_palm = None
                ref_scale = None
                ref_hand_R = None
                ref_robot_pos = None
                ref_robot_quat = None
                ref_robot_R = None
                filtered_pos = None
                filtered_quat = None
                print("Recalibrating on next valid hand frame.")

            if key == ord("r") or key == ord("R"):
                observation, info = env.reset()
                ref_palm = None
                ref_scale = None
                ref_hand_R = None
                ref_robot_pos = None
                ref_robot_quat = None
                ref_robot_R = None
                filtered_pos = None
                filtered_quat = None
                print("Environment reset. Recalibrating on next valid hand frame.")

            if key == ord("t") or key == ord("T"):
                use_rotation = not use_rotation
                filtered_quat = None
                print("use_rotation:", use_rotation)

            time.sleep(1.0 / 240.0)

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()