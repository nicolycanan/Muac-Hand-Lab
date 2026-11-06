"""Preview-only MediaPipe hand tracker without particle effects."""

import cv2
import mediapipe as mp
import numpy as np

from config import CAMERA_BACKEND, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if CAMERA_BACKEND == "DSHOW" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam for hand preview.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    prev = cv2.getTickCount()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for index, landmarks in enumerate(results.multi_hand_landmarks):
                label = "LEFT"
                if results.multi_handedness and index < len(results.multi_handedness):
                    label = results.multi_handedness[index].classification[0].label
                color = (0, 255, 255) if label == "LEFT" else (255, 0, 255)
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    a = connection[0]
                    b = connection[1]
                    pt_a = landmarks.landmark[a]
                    pt_b = landmarks.landmark[b]
                    x1, y1 = int(pt_a.x * frame.shape[1]), int(pt_a.y * frame.shape[0])
                    x2, y2 = int(pt_b.x * frame.shape[1]), int(pt_b.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                for lm in landmarks.landmark:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, color, -1)
                wrist = landmarks.landmark[0]
                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0])
                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        ticks = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (ticks - prev)
        prev = ticks
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("MUAC / HAND PREVIEW", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
