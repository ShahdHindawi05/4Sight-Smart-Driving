from ultralytics import YOLO
from pathlib import Path

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Model loaded successfully!")

image_path = Path(__file__).parent / "test_car.jpg"

if not image_path.exists():
    raise FileNotFoundError("Please put test_car.jpg inside Safe_Following_Distance")

print("Detecting vehicles...")

results = model.predict(
    source=str(image_path),
    classes=[2, 3, 5, 7],
    conf=0.25,
    save=True,
    project=str(Path(__file__).parent / "results"),
    name="vehicle_detection",
    exist_ok=True
)

print("Vehicle detection completed!")

for result in results:
    print("Detected vehicles:", len(result.boxes))

print("Results saved successfully!")
