import copy
import math

import torch
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


class EnergyModel(nn.Module):
    # Maps a 32 x 32 RGB image to one energy score: low for real images, high for others
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 2 * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        self.history = {"loss": [], "real_energy": [], "fake_energy": []}

    def forward(self, x):
        return self.layers(x)


def offset_cosine_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    # diffusion time 0 is a clean image, 1 is almost pure noise
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    angles = start_angle + diffusion_times * (end_angle - start_angle)

    noise_rates = torch.sin(angles)
    signal_rates = torch.cos(angles)
    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer(
            "angular_speeds", 2.0 * math.pi * frequencies.view(1, -1, 1, 1)
        )

    def forward(self, x):
        # (B, 1, 1, 1) noise variance - (B, 2 * num_frequencies, 1, 1)
        return torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)],
            dim=1
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.norm = nn.GroupNorm(8, in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.skip(x)
        x = self.norm(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, width, block_depth=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, width, block_depth=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = self.upsample(x)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    # Predicts the noise in a noisy image, given the noise variance
    def __init__(self, num_channels=3, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)

        # 32 x 32 - 16 x 16 - 8 x 8 - 4 x 4
        self.down1 = DownBlock(32 + embedding_dim, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 96)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        # 4 x 4 - 8 x 8 - 16 x 16 - 32 x 32
        self.up1 = UpBlock(128, 96)
        self.up2 = UpBlock(96, 64)
        self.up3 = UpBlock(64, 32)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        x = self.initial(noisy_images)

        # Copy the embedding to every pixel and stack it with the image features
        embedding = self.embedding(noise_variances)
        embedding = embedding.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, embedding], dim=1)

        skips = []
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


class DiffusionModel(nn.Module):
    def __init__(self, image_size=32, num_channels=3, ema_decay=0.999):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.ema_decay = ema_decay

        # The EMA copy is a smoothed average of the weights and is used for sampling
        self.network = UNet(num_channels)
        self.ema_network = copy.deepcopy(self.network)
        self.ema_network.requires_grad_(False)
        self.ema_steps = 0

        self.history = {"loss": []}

    def denoise(self, noisy_images, noise_rates, signal_rates, use_ema=False):
        network = self.ema_network if use_ema else self.network
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    @torch.no_grad()
    def update_ema(self):
        # A smaller decay at the start, so the random initial weights fade out quickly
        self.ema_steps += 1
        decay = min(self.ema_decay, (1 + self.ema_steps) / (10 + self.ema_steps))

        for ema_param, param in zip(
            self.ema_network.parameters(), self.network.parameters()
        ):
            ema_param.lerp_(param, 1 - decay)


def get_model(model_name):
    if model_name == "GAN":
        return GAN()

    if model_name == "EBM":
        return EnergyModel()

    if model_name == "Diffusion":
        return DiffusionModel()

    raise ValueError(f"Unknown model: {model_name}")
