import torch
import torch.nn as nn


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_traffic_sign_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(model_path, map_location=device, weights_only=False)

    model = TrafficSignCNN(num_classes=bundle["class_count"])
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    return model, bundle, device
