import torch
from torchvision.utils import make_grid

from helper_lib.model import offset_cosine_schedule


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


def seeded_generator(seed=None):
    rng = torch.Generator()
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    return rng


def langevin_sample(model, images, steps, step_size=10.0, noise_std=0.005, rng=None):
    # Move the pixels down the energy gradient, adding a little noise at each step.
    # The weights stay fixed; the gradient is taken with respect to the images.
    model.eval()

    with torch.enable_grad():
        for _ in range(steps):
            with torch.no_grad():
                if rng is None:
                    noise = torch.randn_like(images)
                else:
                    noise = torch.randn(images.shape, generator=rng).to(images.device)
                images = (images + noise_std * noise).clamp(-1, 1)

            images.requires_grad_(True)
            energy = model(images)
            grads, = torch.autograd.grad(energy.sum(), images)

            with torch.no_grad():
                images = (images - step_size * grads.clamp(-0.03, 0.03)).clamp(-1, 1)

    return images.detach()


def generate_ebm_images(model, chains, device='cpu', num_samples=16, steps=60, seed=None):
    rng = seeded_generator(seed)

    # Pick chains saved at the end of training and continue them with the training settings
    index = torch.randperm(len(chains), generator=rng)[:num_samples]
    images = chains[index.to(chains.device)].to(device)
    model = model.to(device)

    images = langevin_sample(
        model, images, steps, step_size=10.0, noise_std=0.005, rng=rng
    )

    return (images.cpu() + 1) / 2


@torch.no_grad()
def generate_diffusion_images(model, device='cpu', num_samples=16, steps=50, seed=None):
    rng = seeded_generator(seed)

    shape = (num_samples, model.num_channels, model.image_size, model.image_size)
    images = torch.randn(shape, generator=rng).to(device)

    model = model.to(device)
    model.ema_network.eval()

    # Reverse diffusion: go from t = 1 (noise) to t = 0 (image) in equal steps
    step_size = 1.0 / steps
    for step in range(steps):
        diffusion_times = torch.full(
            (num_samples, 1, 1, 1), 1.0 - step * step_size, device=device
        )
        noise_rates, signal_rates = offset_cosine_schedule(diffusion_times)
        pred_noises, pred_images = model.denoise(
            images, noise_rates, signal_rates, use_ema=True
        )

        next_noise_rates, next_signal_rates = offset_cosine_schedule(
            diffusion_times - step_size
        )
        images = next_signal_rates * pred_images + next_noise_rates * pred_noises

    return (pred_images.clamp(-1, 1).cpu() + 1) / 2


def show_images(images, title=None):
    import matplotlib.pyplot as plt

    grid = make_grid(images, nrow=min(len(images), 8), pad_value=1)

    plt.figure(figsize=(8, 8 * grid.shape[1] / grid.shape[2]))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
