import torch.nn as nn


class Generator(nn.Module):
    # Layers are defined in the same order as forward()
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim

        # Noise vector - 7 x 7 x 128 feature map
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.reshape = nn.Unflatten(1, (128, 7, 7))

        # 7 x 7 - 14 x 14
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # 14 x 14 - 28 x 28, pixel values in [-1, 1]
        self.deconv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    # Layers are defined in the same order as forward()
    def __init__(self):
        super().__init__()

        # 28 x 28 - 14 x 14
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=4, stride=2, padding=1
        )
        self.act1 = nn.LeakyReLU(0.2)

        # 14 x 14 - 7 x 7
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)

        # One real/fake score per image
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.flatten(x)

        # Returns a logit; sigmoid(logit) is the probability that the image is real.
        # BCEWithLogitsLoss applies the sigmoid inside the loss, which is more stable.
        return self.fc(x)


class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

        # Average losses per epoch, filled in by train_gan
        self.history = {"d_loss": [], "g_loss": []}


def get_model(model_name):
    if model_name == "GAN":
        return GAN()

    raise ValueError(f"Unknown model: {model_name}")
