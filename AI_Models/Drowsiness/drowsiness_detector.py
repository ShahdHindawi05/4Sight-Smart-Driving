
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from pathlib import Path


MODEL_PATH = (
    Path(__file__).parent / "face_landmarker.task"
)

EAR_THRESHOLD = 0.20
PROLONGED_CLOSURE_SECONDS = 1.5

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def calculate_ear(landmarks, eye_indices):
    points = np.array([
        [landmarks[i].x, landmarks[i].y]
        for i in eye_indices
    ])

    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return None

    return (vertical1 + vertical2) / (2 * horizontal)


def analyze_video(video_path, progress_callback=None):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found: {MODEL_PATH}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )

    records = []
    frame_number = 0

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    break

                height, width = frame.shape[:2]

                if width > 640:
                    new_height = int(height * 640 / width)
                    frame = cv2.resize(
                        frame, (640, new_height)
                    )

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    frame_number * 1000 / fps
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                ear = None

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    left = calculate_ear(
                        landmarks, LEFT_EYE
                    )

                    right = calculate_ear(
                        landmarks, RIGHT_EYE
                    )

                    if left is not None and right is not None:
                        ear = (left + right) / 2

                records.append({
                    "Frame": frame_number,
                    "Time": frame_number / fps,
                    "EAR": ear,
                    "Face_Detected": ear is not None,
                    "Eye_Closed": (
                        ear < EAR_THRESHOLD
                        if ear is not None else None
                    )
                })

                frame_number += 1

                if (
                    progress_callback is not None
                    and total_frames > 0
                    and frame_number % 10 == 0
                ):
                    progress_callback(
                        min(frame_number / total_frames, 1.0)
                    )

    finally:
        cap.release()

    if not records:
        raise ValueError("No video frames could be processed.")

    results_df = pd.DataFrame(records)

    # Detect consecutive eye-closure periods.
    events = []
    start_frame = None
    last_frame = None

    for _, row in results_df.iterrows():

        is_closed = (
            row["Face_Detected"]
            and row["Eye_Closed"]
        )

        if is_closed:
            if start_frame is None:
                start_frame = int(row["Frame"])

            last_frame = int(row["Frame"])

        else:
            if start_frame is not None:
                duration = (
                    last_frame - start_frame + 1
                ) / fps

                events.append({
                    "Start Time": round(start_frame / fps, 2),
                    "End Time": round(last_frame / fps, 2),
                    "Duration": round(duration, 2),
                    "Possible Prolonged Closure": (
                        duration >= PROLONGED_CLOSURE_SECONDS
                    )
                })

                start_frame = None
                last_frame = None

    # Save the last event if the video ends with closed eyes.
    if start_frame is not None:
        duration = (last_frame - start_frame + 1) / fps

        events.append({
            "Start Time": round(start_frame / fps, 2),
            "End Time": round(last_frame / fps, 2),
            "Duration": round(duration, 2),
            "Possible Prolonged Closure": (
                duration >= PROLONGED_CLOSURE_SECONDS
            )
        })

    events_df = pd.DataFrame(events)

    detected_frames = int(
        results_df["Face_Detected"].sum()
    )

    summary = {
        "Total Frames": len(results_df),
        "Face Detected Frames": detected_frames,
        "Face Detection Rate": round(
            detected_frames / len(results_df) * 100, 2
        ),
        "Closure Events": len(events_df),
        "Longest Closure": (
            round(events_df["Duration"].max(), 2)
            if not events_df.empty else 0
        )
    }

    return results_df, events_df, summary
