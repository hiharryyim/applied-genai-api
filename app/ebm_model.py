import torch

from app.image_grid import images_to_png
from helper_lib.generator import generate_ebm_images
from helper_lib.model import get_model


class CIFAR10EnergyGenerator:
    def __init__(self, model_path):
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        self.model = get_model("EBM")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Chains are stored as 8-bit pixels; Langevin sampling works in [-1, 1]
        self.chains = checkpoint["chains"].float() / 127.5 - 1

    def generate_png(self, num_samples, steps, seed=None):
        images = generate_ebm_images(
            self.model, self.chains, "cpu", num_samples, steps, seed
        )
        return images_to_png(images)
