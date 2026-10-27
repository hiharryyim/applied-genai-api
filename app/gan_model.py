from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from helper_lib.generator import generate_images
from helper_lib.model import get_model


class MNISTDigitGenerator:
    def __init__(self, model_path):
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        # Only the generator was saved; the discriminator is not needed to create images
        self.model = get_model("GAN")
        self.model.generator.load_state_dict(
            checkpoint["generator_state_dict"]
        )
        self.model.eval()

    def generate_png(self, num_samples, seed=None):
        images = generate_images(self.model, "cpu", num_samples, seed)

        grid = make_grid(images, nrow=min(num_samples, 8), pad_value=1)
        image = to_pil_image(grid).convert("L")

        # Each digit is 28 x 28, so enlarge the grid 4x to make it easier to see.
        # Nearest neighbor keeps the pixels sharp.
        image = image.resize(
            (image.width * 4, image.height * 4),
            resample=Image.Resampling.NEAREST
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
