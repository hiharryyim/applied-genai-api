import torch

from app.image_grid import images_to_png
from helper_lib.generator import generate_diffusion_images
from helper_lib.model import get_model


class CIFAR10DiffusionGenerator:
    def __init__(self, model_path):
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        # Only the EMA network was saved; it is the one used for sampling
        self.model = get_model("Diffusion")
        self.model.ema_network.load_state_dict(checkpoint["ema_state_dict"])
        self.model.eval()

    def generate_png(self, num_samples, steps, seed=None):
        images = generate_diffusion_images(self.model, "cpu", num_samples, steps, seed)
        return images_to_png(images)
