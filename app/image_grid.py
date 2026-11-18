from io import BytesIO

from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def images_to_png(images, scale=4):
    grid = make_grid(images, nrow=min(len(images), 8), pad_value=1)
    image = to_pil_image(grid)

    # CIFAR-10 images are 32 x 32, so enlarge the grid to make it easier to see
    image = image.resize(
        (image.width * scale, image.height * scale),
        resample=Image.Resampling.NEAREST
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
