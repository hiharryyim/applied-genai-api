import torch
from tqdm import tqdm


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
