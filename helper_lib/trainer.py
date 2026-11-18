import torch
from tqdm import tqdm

from helper_lib.generator import langevin_sample
from helper_lib.model import offset_cosine_schedule


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # optimizer is a pair: (generator optimizer, discriminator optimizer)
    gen_optimizer, disc_optimizer = optimizer
    generator = model.generator
    discriminator = model.discriminator

    model.to(device)
    model.train()

    for epoch in range(epochs):
        d_loss_sum = 0.0
        g_loss_sum = 0.0

        for real_images, _ in tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = torch.randn(batch_size, model.z_dim, device=device)
            fake_images = generator(noise)

            # Train the discriminator: real images - 1, generated images - 0
            disc_optimizer.zero_grad()

            d_loss_real = criterion(discriminator(real_images), real_labels)
            # detach() keeps this step from updating the generator
            d_loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            disc_optimizer.step()

            # Train the generator: make the discriminator label its images as real
            gen_optimizer.zero_grad()

            g_loss = criterion(discriminator(fake_images), real_labels)

            g_loss.backward()
            gen_optimizer.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()

        epoch_d_loss = d_loss_sum / len(data_loader)
        epoch_g_loss = g_loss_sum / len(data_loader)

        model.history["d_loss"].append(epoch_d_loss)
        model.history["g_loss"].append(epoch_g_loss)

        print(
            f"D loss: {epoch_d_loss:.4f} | "
            f"G loss: {epoch_g_loss:.4f}"
        )

    return model


class SampleBuffer:
    # Keeps earlier Langevin samples so each training step can continue from them
    def __init__(self, device='cpu', max_size=8192, image_shape=(3, 32, 32)):
        self.device = device
        self.max_size = max_size
        self.image_shape = image_shape
        self.examples = self.random_images(128)

    def random_images(self, n):
        return torch.rand((n, *self.image_shape), device=self.device) * 2 - 1

    def sample(self, model, batch_size, steps, step_size, noise):
        # About 5% of the chains restart from random noise
        n_new = int((torch.rand(batch_size) < 0.05).sum())
        index = torch.randint(len(self.examples), (batch_size - n_new,), device=self.device)
        start = torch.cat([self.random_images(n_new), self.examples[index]])

        samples = langevin_sample(model, start, steps, step_size, noise)

        self.examples = torch.cat([samples, self.examples])[:self.max_size]
        return samples

    @torch.no_grad()
    def select_chains(self, model, n=2048):
        # Keep the n chains with the lowest energy; chains that were just restarted are still noise
        energies = torch.cat([model(chunk) for chunk in self.examples.split(1024)])
        index = energies.squeeze(1).argsort()[:n]
        return self.examples[index]


def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10,
              alpha=0.1, steps=60, step_size=10.0, noise=0.005):
    model.to(device)
    buffer = SampleBuffer(device)

    for epoch in range(epochs):
        loss_sum = 0.0
        real_sum = 0.0
        fake_sum = 0.0

        for real_images, _ in tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            real_images = real_images.to(device)
            real_images = (real_images + noise * torch.randn_like(real_images)).clamp(-1, 1)
            batch_size = real_images.size(0)

            fake_images = buffer.sample(model, batch_size, steps, step_size, noise)

            model.train()
            energies = model(torch.cat([real_images, fake_images]))
            real_energy, fake_energy = energies.split(batch_size)

            # Push real energies down and sample energies up;
            # the squared term keeps the scores from drifting too far
            cdiv_loss = real_energy.mean() - fake_energy.mean()
            reg_loss = alpha * (real_energy.pow(2).mean() + fake_energy.pow(2).mean())
            loss = cdiv_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            loss_sum += loss.item()
            real_sum += real_energy.mean().item()
            fake_sum += fake_energy.mean().item()

        n = len(data_loader)
        model.history["loss"].append(loss_sum / n)
        model.history["real_energy"].append(real_sum / n)
        model.history["fake_energy"].append(fake_sum / n)

        print(
            f"Loss: {loss_sum / n:.4f} | "
            f"Real energy: {real_sum / n:.4f} | "
            f"Sample energy: {fake_sum / n:.4f}"
        )

    # The buffer holds the model's persistent chains, which are used again for generation
    return model, buffer


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)

    for epoch in range(epochs):
        model.network.train()
        loss_sum = 0.0

        for images, _ in tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device)
            noises = torch.randn_like(images)

            # A random noise level for each image
            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=device)
            noise_rates, signal_rates = offset_cosine_schedule(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            pred_noises, _ = model.denoise(noisy_images, noise_rates, signal_rates)
            loss = criterion(pred_noises, noises)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()

            loss_sum += loss.item()

        epoch_loss = loss_sum / len(data_loader)
        model.history["loss"].append(epoch_loss)

        print(f"Loss: {epoch_loss:.4f}")

    return model
