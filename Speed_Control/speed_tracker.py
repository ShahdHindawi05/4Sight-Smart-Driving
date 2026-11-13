
import cv2
import pandas as pd
import math
import json
from pathlib import Path
from ultralytics import YOLO


def analyze_video(video_path, output_dir=None):

    video_path = Path(video_path)

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / "tracked_video.mp4"
    output_csv = output_dir / "vehicle_tracking.csv"

    model_path = Path(__file__).resolve().parent.parent / "yolov8n.pt"
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not fps or fps <= 0:
        raise ValueError("Could not read video FPS.")

    writer = None
    records = []
    previous_positions = {}
    frame_number = 0

    results = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        classes=[2],
        conf=0.25,
        tracker="bytetrack.yaml",
        verbose=False
    )

    for result in results:

        frame_number += 1
        annotated_frame = result.plot()

        if writer is None:
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        writer.write(annotated_frame)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy()
        else:
            track_ids = [None] * len(boxes)

        current_time = frame_number / fps

        for box, confidence, track_id in zip(
            boxes, confidences, track_ids
        ):

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            pixel_speed = None

            if track_id is not None:
                track_id = int(track_id)

                if track_id in previous_positions:
                    old_x, old_y, old_time = previous_positions[track_id]

                    delta_time = current_time - old_time

                    if delta_time > 0:
                        distance_pixels = math.sqrt(
                            (center_x - old_x) ** 2 +
                            (center_y - old_y) ** 2
                        )

                        pixel_speed = distance_pixels / delta_time

                previous_positions[track_id] = (
                    center_x, center_y, current_time
                )

            records.append({
                "Frame": frame_number,
                "Time": current_time,
                "Track_ID": track_id,
                "Center_X": center_x,
                "Center_Y": center_y,
                "Box_Width": x2 - x1,
                "Box_Height": y2 - y1,
                "Pixel_Speed_Per_Second": pixel_speed,
                "Confidence": float(confidence)
            })

    if writer is not None:
        writer.release()

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No cars were detected in this video.")

    df.to_csv(output_csv, index=False)

    calibration_path = Path(__file__).parent / "road_calibration.json"

    calibration_status = "Not available"

    if calibration_path.exists():
        with open(calibration_path, "r") as file:
            calibration = json.load(file)

        distance = calibration.get("road_distance_meters", 0)

        if distance > 0:
            calibration_status = (
                "Experimental calibration only - "
                "real-world distance has not been verified."
            )

    return {
        "tracking_data": df,
        "video_path": str(output_video),
        "csv_path": str(output_csv),
        "fps": fps,
        "frames_processed": frame_number,
        "calibration_status": calibration_status
    }
