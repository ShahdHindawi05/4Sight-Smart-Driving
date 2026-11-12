
from ultralytics import YOLO
from pathlib import Path
import cv2

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")
print("Model loaded successfully!")

root = Path(__file__).parent
video_path = root / "8358-208052058_medium.mp4"

if not video_path.exists():
    raise FileNotFoundError("Video file not found!")

cap = cv2.VideoCapture(str(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_dir = root / "results"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "safe_distance_warning.mp4"

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Statistics
total_detections = 0
caution_count = 0
warning_count = 0
frame_number = 0

print("Analyzing video...")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.predict(
        frame,
        classes=[2, 3, 5, 7],
        conf=0.25,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            total_detections += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_height = y2 - y1

            if box_height > height * 0.35:
                label = "WARNING: VERY CLOSE"
                color = (0, 0, 255)
                warning_count += 1

            elif box_height > height * 0.20:
                label = "CAUTION: GETTING CLOSE"
                color = (0, 165, 255)
                caution_count += 1

            else:
                label = "Vehicle detected"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    writer.write(frame)

    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames")

cap.release()
writer.release()

print("\n========== PROJECT STATISTICS ==========")
print("Total frames processed:", frame_number)
print("Total vehicle detections:", total_detections)
print("Caution detections:", caution_count)
print("Warning detections:", warning_count)
print("========================================")

print("\nVideo saved successfully:")
print(output_path)

