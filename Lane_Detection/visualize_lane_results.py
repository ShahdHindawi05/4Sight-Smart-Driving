import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# إعدادات الجهاز
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.70

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(
    BASE_DIR, "Datasets", "TuSimple", "tusimple",
    "tusimple", "training", "frames"
)
MASKS_DIR = os.path.join(
    BASE_DIR, "Datasets", "TuSimple", "tusimple",
    "tusimple", "training", "lane-masks"
)
CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "AI_Models", "Lane_Detection", "lane_unet_best.pth"
)
SPLIT_FILE = os.path.join(BASE_DIR, "lane_data_split_by_clip.csv")


# بناء النموذج نفسه الذي استخدمناه بالتدريب
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
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.enc4 = ConvBlock(64, 128)
        self.bottleneck = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

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

        d4 = F.interpolate(
            b, size=e4.shape[2:], mode="bilinear", align_corners=False
        )
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(
            d4, size=e3.shape[2:], mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(
            d3, size=e2.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2, size=e1.shape[2:], mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


# تحميل النموذج
print("Loading trained lane detection model...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = UNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

IMAGE_WIDTH = checkpoint["image_width"]
IMAGE_HEIGHT = checkpoint["image_height"]

print("Model loaded successfully!")


# قراءة صور الاختبار
df = pd.read_csv(SPLIT_FILE)
test_df = df[df["split"] == "test"].reset_index(drop=True)

# اختيار 5 صور من الاختبار
selected = test_df.sample(
    n=min(5, len(test_df)),
    random_state=42
)

output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "visualized_results"
)
os.makedirs(output_dir, exist_ok=True)


# عرض النتائج وحفظها
for i, row in selected.iterrows():

    filename = row["filename"]

    image_path = os.path.join(FRAMES_DIR, filename)
    mask_path = os.path.join(MASKS_DIR, filename)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Could not load:", filename)
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        image_rgb,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    input_tensor = torch.tensor(
        resized.transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0) / 255.0

    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        prediction = torch.sigmoid(model(input_tensor))
        prediction = prediction.squeeze().cpu().numpy()

    predicted_mask = (prediction >= THRESHOLD).astype(np.uint8)

    predicted_mask = cv2.resize(
        predicted_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    true_mask = (mask > 127).astype(np.uint8)

    # إنشاء صورة Overlay
    overlay = image_rgb.copy()
    overlay[predicted_mask == 1] = (
        0.5 * overlay[predicted_mask == 1]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # عرض الصور الثلاث
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Road Image")

    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(overlay)
    axes[2].set_title("Model Prediction Overlay (Red)")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(filename)
    plt.tight_layout()

    save_path = os.path.join(
        output_dir, f"result_{i+1}.png"
    )

    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Saved:", save_path)

print("\nAll visualizations completed!")
print("Results folder:", output_dir)