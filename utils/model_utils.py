import torch

def get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, noise = None):
    # Sample noise that we'll add to the latents
    if noise is None:
        noise = torch.randn_like(latents, device=latents.device)
    b_size = latents.shape[0]
    min_timestep = 0
    max_timestep = noise_scheduler.config.num_train_timesteps
    timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)
    timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return noise, noisy_latents, timesteps

