
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ==========================================
# 1. Paths and settings
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    ROOT / "Datasets" / "TuSimple" /
    "tusimple" / "tusimple" /
    "training" / "frames" / "0601_1494452385593783358.jpg"
)

MODEL_PATH = (
    ROOT / "AI_Models" / "Lane_Detection" /
    "lane_unet_best.pth"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parent /
    "lane_prediction.png"
)

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180


# ==========================================
# 2. ConvBlock - Same as training
# ==========================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ==========================================
# 3. U-Net - Exact training architecture
# ==========================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.enc4 = ConvBlock(64, 128)

        self.bottleneck = ConvBlock(128, 256)

        self.dec4 = ConvBlock(256 + 128, 128)
        self.dec3 = ConvBlock(128 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = nn.functional.interpolate(
            b, size=e4.shape[2:],
            mode="bilinear", align_corners=False
        )

        d4 = self.dec4(
            torch.cat([d4, e4], dim=1)
        )

        d3 = nn.functional.interpolate(
            d4, size=e3.shape[2:],
            mode="bilinear", align_corners=False
        )

        d3 = self.dec3(
            torch.cat([d3, e3], dim=1)
        )

        d2 = nn.functional.interpolate(
            d3, size=e2.shape[2:],
            mode="bilinear", align_corners=False
        )

        d2 = self.dec2(
            torch.cat([d2, e2], dim=1)
        )

        d1 = nn.functional.interpolate(
            d2, size=e1.shape[2:],
            mode="bilinear", align_corners=False
        )

        d1 = self.dec1(
            torch.cat([d1, e1], dim=1)
        )

        return self.output(d1)


# ==========================================
# 4. Main testing function
# ==========================================

def main():

    print("Starting Lane Detection Testing...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Check files

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Load image

    image = Image.open(IMAGE_PATH).convert("RGB")

    original = np.array(image)

    resized = image.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        resized, dtype=np.float32
    ) / 255.0

    input_tensor = torch.from_numpy(
        image_array
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    # Load model checkpoint

    model = UNet().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully!")
    print(
        "Best Validation Dice:",
        checkpoint["best_val_dice"]
    )

    # Prediction

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.sigmoid(outputs)

        prediction = (
            probabilities.squeeze().cpu().numpy()
        )

    # Convert prediction to original size

    prediction_image = Image.fromarray(
        (prediction * 255).astype(np.uint8)
    )

    prediction_image = prediction_image.resize(
        (original.shape[1], original.shape[0]),
        Image.Resampling.BILINEAR
    )

    mask = np.array(prediction_image) > 127

    # Create overlay

    overlay = original.copy()

    overlay[mask] = (
        0.5 * overlay[mask] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Display results

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Road Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Lane Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Lane Detection Result")
    plt.axis("off")

    plt.tight_layout()

    # Save image

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\nTesting completed successfully!")
    print("Result saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
