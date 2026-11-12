
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. إعداد المسارات والإعدادات
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "Datasets" / "TuSimple" /
    "tusimple" / "tusimple" / "training"
)

FRAMES_DIR = DATA_DIR / "frames"
MASKS_DIR = DATA_DIR / "lane-masks"

SPLIT_FILE = ROOT / "lane_data_split_by_clip.csv"

OUTPUT_DIR = ROOT / "AI_Models" / "Lane_Detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. تحميل بيانات الصور والـ Masks
# ==========================================

class LaneDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        filename = self.dataframe.iloc[index]["filename"]

        image_path = FRAMES_DIR / filename
        mask_path = MASKS_DIR / filename

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.NEAREST
        )

        image = np.array(image, dtype=np.float32) / 255.0

        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# ==========================================
# 3. نموذج U-Net
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
            b, size=e4.shape[2:], mode="bilinear",
            align_corners=False
        )
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = nn.functional.interpolate(
            d4, size=e3.shape[2:], mode="bilinear",
            align_corners=False
        )
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = nn.functional.interpolate(
            d3, size=e2.shape[2:], mode="bilinear",
            align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = nn.functional.interpolate(
            d2, size=e1.shape[2:], mode="bilinear",
            align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


# ==========================================
# 4. حساب Dice Score
# ==========================================

def dice_score(predictions, targets):

    predictions = (torch.sigmoid(predictions) > 0.5).float()

    intersection = (predictions * targets).sum(dim=(1, 2, 3))

    total = (
        predictions.sum(dim=(1, 2, 3)) +
        targets.sum(dim=(1, 2, 3))
    )

    score = (2 * intersection + 1e-7) / (total + 1e-7)

    return score.mean().item()


# ==========================================
# 5. تجهيز البيانات
# ==========================================

def main():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    if not FRAMES_DIR.exists() or not MASKS_DIR.exists():
        raise FileNotFoundError(
            "Frames or lane-masks folder was not found."
        )

    df = pd.read_csv(SPLIT_FILE)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    print("\nDataset sizes:")
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("One of the dataset splits is empty.")

    train_loader = DataLoader(
        LaneDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        LaneDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        LaneDataset(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    # ======================================
    # 6. إنشاء النموذج والتدريب
    # ======================================

    model = UNet().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([10.0], device=DEVICE)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_dice = 0.0
    best_model_path = OUTPUT_DIR / "lane_unet_best.pth"

    print("\nStarting Lane Detection Training...\n")

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            bce = criterion(outputs, masks)

            probabilities = torch.sigmoid(outputs)

            intersection = (probabilities * masks).sum(
                dim=(1, 2, 3)
            )

            denominator = (
                probabilities.sum(dim=(1, 2, 3)) +
                masks.sum(dim=(1, 2, 3))
            )

            dice_loss = 1 - (
                (2 * intersection + 1e-7) /
                (denominator + 1e-7)
            ).mean()

            loss = bce + dice_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation

        model.eval()

        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                bce = criterion(outputs, masks)

                probabilities = torch.sigmoid(outputs)

                intersection = (probabilities * masks).sum(
                    dim=(1, 2, 3)
                )

                denominator = (
                    probabilities.sum(dim=(1, 2, 3)) +
                    masks.sum(dim=(1, 2, 3))
                )

                dice_loss = 1 - (
                    (2 * intersection + 1e-7) /
                    (denominator + 1e-7)
                ).mean()

                val_loss += (bce + dice_loss).item()
                val_dice += dice_score(outputs, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f} "
            f"Validation Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_width": IMAGE_WIDTH,
                    "image_height": IMAGE_HEIGHT,
                    "best_val_dice": best_val_dice
                },
                best_model_path
            )

            print("Best model saved!")


    # ======================================
    # 7. اختبار أفضل نموذج
    # ======================================

    print("\nLoading best model for testing...")

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dice = 0.0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            test_dice += dice_score(outputs, masks)

    test_dice /= len(test_loader)

    print("\n========== Final Results ==========")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}")
    print("Model saved at:", best_model_path)
    print("Training completed!")


if __name__ == "__main__":
    main()
