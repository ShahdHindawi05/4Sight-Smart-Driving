
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# مسار المشروع الرئيسي
ROOT = Path(__file__).resolve().parent.parent

# نموذج اكتشاف الوجه
MODEL_PATH = ROOT / "AI_Models" / "Drowsiness" / "face_landmarker.task"

# نقاط الوجه المستخدمة لتقدير وضعية الرأس
LANDMARKS = [1, 152, 33, 263, 61, 291]

# نموذج تقريبي ثلاثي الأبعاد لنقاط الوجه
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 63.6, -12.5],
    [-43.3, -32.7, -26.0],
    [43.3, -32.7, -26.0],
    [-28.9, 28.9, -24.1],
    [28.9, 28.9, -24.1]
], dtype=np.float64)


def analyze_video(video_path, output_dir=None):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face model not found: {MODEL_PATH}"
        )

    # مجلد حفظ النتائج
    if output_dir is None:
        output_dir = ROOT / "Mirror_Usage"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # أسماء ملفات النتائج
    OUTPUT_PATH = output_dir / "mirror_head_pose.csv"
    OUTPUT_VIDEO = output_dir / "mirror_head_pose_visual.mp4"
    EVENTS_PATH = output_dir / "mirror_events.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera_matrix = np.array([
        [width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    results = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        1,
        (width, height)
    )

    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:

            second = 0

            while True:

                # قراءة إطار واحد كل ثانية
                cap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    second * 1000
                )

                success, frame = cap.read()

                if not success:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect_for_video(
                    image,
                    second * 1000
                )

                row = {
                    "time_seconds": second,
                    "face_detected": False,
                    "yaw": np.nan,
                    "pitch": np.nan,
                    "roll": np.nan
                }

                if result.face_landmarks:

                    face = result.face_landmarks[0]

                    image_points = np.array([
                        [face[i].x * width, face[i].y * height]
                        for i in LANDMARKS
                    ], dtype=np.float64)

                    success, rotation_vector, translation_vector = (
                        cv2.solvePnP(
                            MODEL_POINTS,
                            image_points,
                            camera_matrix,
                            dist_coeffs
                        )
                    )

                    if success:

                        rotation_matrix, _ = cv2.Rodrigues(
                            rotation_vector
                        )

                        axis_length = 50

                        axis_points_3d = np.float64([
                            [axis_length, 0, 0],
                            [0, axis_length, 0],
                            [0, 0, axis_length]
                        ])

                        axis_points_2d, _ = cv2.projectPoints(
                            axis_points_3d,
                            rotation_vector,
                            translation_vector,
                            camera_matrix,
                            dist_coeffs
                        )

                        nose = tuple(
                            image_points[0].astype(int)
                        )

                        x_axis = tuple(
                            axis_points_2d[0].ravel().astype(int)
                        )

                        y_axis = tuple(
                            axis_points_2d[1].ravel().astype(int)
                        )

                        z_axis = tuple(
                            axis_points_2d[2].ravel().astype(int)
                        )

                        cv2.line(
                            frame, nose, x_axis,
                            (0, 0, 255), 3
                        )

                        cv2.line(
                            frame, nose, y_axis,
                            (0, 255, 0), 3
                        )

                        cv2.line(
                            frame, nose, z_axis,
                            (255, 0, 0), 3
                        )

                        angles = cv2.RQDecomp3x3(
                            rotation_matrix
                        )[0]

                        row["face_detected"] = True
                        row["pitch"] = angles[0]
                        row["yaw"] = angles[1]
                        row["roll"] = (
                            (angles[2] + 90) % 180
                        ) - 90

                results.append(row)

                video_writer.write(frame)

                second += 1

    finally:
        cap.release()
        video_writer.release()

    # تحويل النتائج إلى DataFrame
    df = pd.DataFrame(results)

    if df.empty:
        raise RuntimeError("No frames could be read from the video.")

    # تنعيم زوايا الرأس
    df["yaw_smooth"] = (
        df["yaw"].rolling(
            window=3,
            center=True,
            min_periods=2
        ).median()
    )

    df["pitch_smooth"] = (
        df["pitch"].rolling(
            window=3,
            center=True,
            min_periods=2
        ).median()
    )

    df["roll_smooth"] = (
        df["roll"].rolling(
            window=3,
            center=True,
            min_periods=2
        ).median()
    )

    # العينات التي لم يتم اكتشاف الوجه فيها
    missing_face = df["face_detected"] == False

    df.loc[
        missing_face,
        ["yaw_smooth", "pitch_smooth", "roll_smooth"]
    ] = np.nan

    # أول 3 ثوانٍ تستخدم كمرجع للنظر للأمام
    baseline_data = df.loc[
        (df["time_seconds"] < 3) &
        (df["face_detected"]),
        "yaw_smooth"
    ].dropna()

    if baseline_data.empty:
        raise RuntimeError(
            "Could not establish a forward-looking baseline. "
            "Make sure the driver's face is visible at the beginning."
        )

    baseline = baseline_data.median()

    # حساب انحراف الرأس عن وضعية النظر للأمام
    df["yaw_relative"] = (
        df["yaw_smooth"] - baseline
    )

    # عتبة الالتفات الجانبي
    THRESHOLD = 25

    df["side_looking"] = (
        df["yaw_relative"].abs() >= THRESHOLD
    )

    # تجاهل العينات التي لم يتم اكتشاف الوجه فيها
    df.loc[
        missing_face,
        "side_looking"
    ] = False

    # تجميع الالتفاتات المتقاربة
    events_data = df[
        df["side_looking"]
    ].copy()

    if not events_data.empty:

        events_data["group"] = (
            events_data["time_seconds"]
            .diff()
            .fillna(999)
            .gt(2)
            .cumsum()
        )

        events_df = events_data.groupby("group").agg(
            start_time=("time_seconds", "min"),
            end_time=("time_seconds", "max"),
            samples=("time_seconds", "count")
        ).reset_index(drop=True)

        events_df["duration_seconds"] = (
            events_df["end_time"]
            - events_df["start_time"] + 1
        )

    else:

        events_df = pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "samples",
                "duration_seconds"
            ]
        )

    # حفظ النتائج
    df.to_csv(OUTPUT_PATH, index=False)
    events_df.to_csv(EVENTS_PATH, index=False)

    print("\nMirror Usage Analysis Completed!")
    print("Video:", video_path.name)
    print("Samples:", len(df))
    print("Detected faces:", int(df["face_detected"].sum()))
    print("Forward baseline:", round(baseline, 2))
    print("Possible side-looking samples:", int(df["side_looking"].sum()))
    print("Possible events:", len(events_df))

    print("\nMirror Usage Report")
    print(events_df.to_string(index=False))

    return df, events_df, str(OUTPUT_VIDEO)
