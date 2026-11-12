
from pathlib import Path
import pandas as pd
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT / "Lane_Detection"))

from test_lane import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180

CHECKPOINT_PATH = ROOT / "AI_Models" / "Lane_Detection" / "lane_unet_best.pth"

FRAMES_DIR = (
    ROOT / "Datasets" / "TuSimple" /
    "tusimple" / "tusimple" /
    "training" / "frames"
)

MASKS_DIR = (
    ROOT / "Datasets" / "TuSimple" /
    "tusimple" / "tusimple" /
    "training" / "lane-masks"
)

RESULTS_PATH = ROOT / "Lane_Detection" / "lane_test_results.csv"

OUTPUT_DIR = ROOT / "Lane_Detection" / "worst_images"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading results...")
df = pd.read_csv(RESULTS_PATH)
worst_images = df.nsmallest(10, "dice_score")

print("Loading trained model...")
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = UNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Inspecting worst 10 images...")

for index, row in enumerate(worst_images.itertuples(), start=1):

    filename = row.filename

    image_path = FRAMES_DIR / filename
    mask_path = MASKS_DIR / filename

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Could not read:", filename)
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(
        image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    input_tensor = torch.from_numpy(
        resized_image.transpose(2, 0, 1)
    ).float() / 255.0

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = torch.sigmoid(model(input_tensor))
        prediction = (prediction > 0.5).float()

    pred_mask = prediction.squeeze().cpu().numpy()

    pred_mask = cv2.resize(
        pred_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    true_mask = (mask > 127).astype(np.uint8)

    overlay = image_rgb.copy()
    overlay[pred_mask == 1] = (
        255, 0, 0
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Road Image")

    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction | Dice: {row.dice_score:.3f}"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{index}. {filename}")
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"worst_{index}.png"
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

    print(f"Saved: {save_path}")

print("\nDone! Images saved in:", OUTPUT_DIR)
