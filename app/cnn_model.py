from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CIFAR10Classifier:
    def __init__(self, model_path):
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        self.class_names = checkpoint["class_names"]

        self.model = AssignmentCNN()
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

        # Match the preprocessing used during training
        self.transform = transforms.Compose([
            transforms.Resize(
                tuple(checkpoint["image_size"])
            ),
            transforms.ToTensor()
        ])

    def predict(self, image_bytes):
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            inputs = self.transform(image).unsqueeze(0)

        with torch.inference_mode():
            scores = self.model(inputs)
            probabilities = torch.softmax(scores, dim=1)[0]

        confidence, index = probabilities.max(dim=0)
        class_id = index.item()

        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": confidence.item()
        }
