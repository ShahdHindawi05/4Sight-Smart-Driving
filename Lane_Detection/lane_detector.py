import torch
import torch.nn as nn
import numpy as np
from PIL import Image

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
        self.dec4 = ConvBlock(256 + 128, 128)
        self.dec3 = ConvBlock(128 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = nn.functional.interpolate(b, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = nn.functional.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = nn.functional.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = nn.functional.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.output(d1)

def load_lane_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device

def predict_lane(model, device, image, threshold=0.70):
    original = image.convert("RGB")
    resized = original.resize((320, 180))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    probability_image = Image.fromarray(
        (probabilities * 255).astype(np.uint8)
    ).resize(original.size, Image.Resampling.BILINEAR)

    mask = np.asarray(probability_image) > int(threshold * 255)
    original_array = np.asarray(original).copy()
    overlay = original_array.copy()
    overlay[mask] = (
        0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return Image.fromarray(mask.astype(np.uint8) * 255), Image.fromarray(overlay)
