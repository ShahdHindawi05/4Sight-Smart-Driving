from ultralytics import YOLO
from pathlib import Path

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")
print("Model loaded successfully!")

root = Path(__file__).parent
video_path = root / "8358-208052058_medium.mp4"

if not video_path.exists():
    raise FileNotFoundError("Video file not found!")

print("Starting vehicle detection on video...")

results = model.predict(
    source=str(video_path),
    classes=[2, 3, 5, 7],
    conf=0.25,
    save=True,
    project=str(root / "results"),
    name="video_detection",
    exist_ok=True,
    vid_stride=3
)

print("Video processing completed!")
print("Results saved in:")
print(root / "results" / "video_detection")
