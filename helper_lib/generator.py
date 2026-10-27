import torch
from torchvision.utils import make_grid


def generate_images(model, device='cpu', num_samples=10, seed=None):
    # Sample points in the latent space and run the generator
    rng = torch.Generator()
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)

    noise = torch.randn(num_samples, model.z_dim, generator=rng).to(device)

    generator = model.generator.to(device)
    generator.eval()

    with torch.no_grad():
        images = generator(noise)

    # Tanh output [-1, 1] - pixel values [0, 1]
    return (images.cpu() + 1) / 2


def generate_samples(model, device, num_samples=10):
    # matplotlib is only needed for plotting in the notebook, not in the API
    import matplotlib.pyplot as plt

    images = generate_images(model, device, num_samples)
    grid = make_grid(images, nrow=min(num_samples, 8), pad_value=1)

    plt.figure(figsize=(8, 8 * grid.shape[1] / grid.shape[2]))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
