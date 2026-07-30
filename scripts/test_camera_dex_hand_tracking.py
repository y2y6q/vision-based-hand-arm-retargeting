import cv2
import numpy as np
import mediapipe as mp


def draw_landmarks(frame, keypoint_2d):
    # MediaPipe hand landmark connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for i, j in connections:
        p1 = tuple(keypoint_2d[i].astype(int))
        p2 = tuple(keypoint_2d[j].astype(int))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    for idx, point in enumerate(keypoint_2d):
        x, y = point.astype(int)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(idx),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )


def main():
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0")

    print("Camera opened.")
    print("Show your hand to the camera.")
    print("Press Q to exit.")

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            keypoint_2d = []
            joint_pos = []

            for lm in hand_landmarks.landmark:
                x_px = lm.x * w
                y_px = lm.y * h

                keypoint_2d.append([x_px, y_px])

                # 这个 joint_pos 是后面接 dex-retargeting / arm-follow 的基础格式
                # x, y, z 都是 MediaPipe normalized coordinates
                joint_pos.append([lm.x, lm.y, lm.z])

            keypoint_2d = np.asarray(keypoint_2d, dtype=np.float32)
            joint_pos = np.asarray(joint_pos, dtype=np.float32)

            draw_landmarks(frame, keypoint_2d)

            wrist = joint_pos[0]
            index_tip = joint_pos[8]
            middle_tip = joint_pos[12]
            palm_center = np.mean(
                joint_pos[[0, 5, 9, 13, 17]],
                axis=0,
            )

            if frame_id % 10 == 0:
                print(
                    "wrist=",
                    np.round(wrist, 4),
                    " palm=",
                    np.round(palm_center, 4),
                    " index_tip=",
                    np.round(index_tip, 4),
                    " middle_tip=",
                    np.round(middle_tip, 4),
                )

            cv2.putText(
                frame,
                f"wrist: {wrist[0]:.2f}, {wrist[1]:.2f}, {wrist[2]:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

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

        cv2.imshow("Camera Hand Tracking - MediaPipe compatible with Dex pipeline", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

        frame_id += 1

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()