
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT / "Datasets" / "TuSimple" /
    "tusimple" / "tusimple" / "training"
)

SPLIT_FILE = ROOT / "lane_data_split_by_clip.csv"

MODEL_PATH = (
    ROOT / "AI_Models" / "Lane_Detection" /
    "lane_unet_best.pth"
)

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180

THRESHOLD = 0.70

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
print("Threshold:", THRESHOLD)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.enc4 = ConvBlock(64, 128)

        self.bottleneck = ConvBlock(128, 256)

        self.dec4 = ConvBlock(384, 128)
        self.dec3 = ConvBlock(192, 64)
        self.dec2 = ConvBlock(96, 32)
        self.dec1 = ConvBlock(48, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = F.interpolate(
            b, size=e4.shape[2:],
            mode="bilinear", align_corners=False
        )
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(
            d4, size=e3.shape[2:],
            mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(
            d3, size=e2.shape[2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2, size=e1.shape[2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = UNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")


df = pd.read_csv(SPLIT_FILE)
test_df = df[df["split"] == "test"].copy()

print("Test images:", len(test_df))

dice_scores = []
evaluated_filenames = []
missing_files = []

print("\nEvaluating test images...")

for index, row in enumerate(test_df.itertuples(), start=1):

    image_path = DATA_DIR / "frames" / row.filename
    mask_path = DATA_DIR / "lane-masks" / row.filename

    if not image_path.exists() or not mask_path.exists():
        missing_files.append(row.filename)
        continue

    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(image, dtype=np.float32) / 255.0

    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.NEAREST
    )

    true_mask = np.array(mask) > 127

    with torch.no_grad():
        probabilities = torch.sigmoid(model(image_tensor))
        predicted_mask = (
            probabilities.squeeze().cpu().numpy() > THRESHOLD
        )

    intersection = np.logical_and(
        predicted_mask, true_mask
    ).sum()

    total_pixels = predicted_mask.sum() + true_mask.sum()

    dice = (
        2 * intersection + 1e-7
    ) / (
        total_pixels + 1e-7
    )

    dice_scores.append(dice)
    evaluated_filenames.append(row.filename)

    if index % 50 == 0:
        print(f"Processed {index}/{len(test_df)} images")


if len(dice_scores) == 0:
    raise ValueError("No test images were evaluated.")


mean_dice = np.mean(dice_scores)

results = pd.DataFrame({
    "filename": evaluated_filenames,
    "dice_score": dice_scores
})

output_path = (
    ROOT / "Lane_Detection" /
    "lane_test_results_threshold_070.csv"
)

results.to_csv(output_path, index=False)

print("\n========== FINAL TEST RESULTS ==========")
print("Threshold:", THRESHOLD)
print("Total test images:", len(test_df))
print("Successfully evaluated:", len(dice_scores))
print("Missing files:", len(missing_files))
print(f"Average Dice: {mean_dice:.4f}")
print(f"Average Dice Percentage: {mean_dice * 100:.2f}%")

print("\n========== COMPARISON ==========")
print("Original Test Dice (Threshold 0.50): 71.72%")
print(f"New Test Dice (Threshold 0.70): {mean_dice * 100:.2f}%")

print("\nResults saved to:", output_path)
