from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolov8n.pt"


def analyze_video(video_path, output_dir=None):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    if "cell phone" not in model.names.values():
        raise ValueError(
            "The YOLO model does not contain the cell phone class."
        )

    phone_class_id = next(
        class_id
        for class_id, class_name in model.names.items()
        if class_name == "cell phone"
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = output_dir / "phone_detection_raw.mp4"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video.")

    events = []

    review_dir = output_dir / "phone_review_frames"
    review_dir.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    frame_interval = 5

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_number % frame_interval == 0:

                results = model.predict(
                    frame,
                    conf=0.70,
                    verbose=False
                )

                for box in results[0].boxes:

                    class_id = int(box.cls[0])

                    if class_id != phone_class_id:
                        continue

                    confidence = float(box.conf[0])

                    second = int(frame_number / fps)


                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "PHONE DETECTED - REVIEW",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    image_name = f"frame_{frame_number}_phone.jpg"
                    image_path = review_dir / image_name

                    cv2.imwrite(str(image_path), frame)


                    events.append({
                        "Time (sec)": second,
                        "Detected Object": "تم اكتشاف هاتف",
                        "Confidence": round(confidence, 2),
                        "Status": "Needs Review",
                        "Review Image": str(image_path)
                    })
            writer.write(frame)
            frame_number += 1

    
    finally:
            cap.release()
    writer.release()

    report = pd.DataFrame(events)

    csv_path = output_dir / "phone_events.csv"
    report.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    return report, output_video, csv_path
