import torch

def encode_sequence_with_vae(vae, frames):
    # frames: (batch, T, 3, H, W)
    latent_sequence = []
    for t in range(frames.shape[1]):
        frame_latent = vae.encode_image(frames[:, t])
        latent_sequence.append(frame_latent)
    # Stack across time => shape (batch, T, latent_channels, h, w)
    return torch.stack(latent_sequence, dim=1)

def decode_sequence_with_vae(vae, latent_seq):
    # latent_seq: (batch, T, latent_channels, h, w)
    decoded_frames = []
    for t in range(latent_seq.shape[1]):
        decoded_frame = vae.decode_latents(latent_seq[:, t])
        decoded_frames.append(decoded_frame)
    return torch.stack(decoded_frames, dim=1)  # (batch, T, 3, H, W)

def make_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2):
    """
    Creates a linear schedule for beta from beta_start to beta_end.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)

def q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Forward diffusion: produce x_t from x_0 by adding noise.

    Args:
        x0 (torch.Tensor): Original latent sequences, shape (B, T, C, H, W)
        t (torch.Tensor): Timesteps for each element, shape (B, T)
        sqrt_alphas_cumprod (torch.Tensor): Precomputed sqrt(alpha_cumprod), shape (num_timesteps,)
        sqrt_one_minus_alphas_cumprod (torch.Tensor): Precomputed sqrt(1 - alpha_cumprod), shape (num_timesteps,)

    Returns:
        torch.Tensor: Noisy latent sequences, shape (B, T, C, H, W)
        torch.Tensor: Noise added, shape (B, T, C, H, W)
    """
    B, T, C, H, W = x0.shape
    noise = torch.randn_like(x0)

    t = t.cpu()
    # Gather coefficients for each timestep
    alphas_t = sqrt_alphas_cumprod[t].view(B, T, 1, 1, 1)
    one_minus_alphas_t = sqrt_one_minus_alphas_cumprod[t].view(B, T, 1, 1, 1)
    alphas_t = alphas_t.to(x0.device)
    one_minus_alphas_t = one_minus_alphas_t.to(x0.device)
    
    # Add noise to the original latent sequences
    x_t = alphas_t * x0 + one_minus_alphas_t * noise
    return x_t, noise

def p_sample(model, x_t, t, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, betas_t):
    """
    Single sampling step: compute x_{t-1} using the model (UNet + Transformer).

    Args:
        model (nn.Module): The diffusion model.
        x_t (torch.Tensor): Current noisy latent sequences, shape (B, T, C, H, W)
        t (torch.Tensor): Current timesteps, shape (B, T)
        sqrt_recip_alphas_cumprod (torch.Tensor): Precomputed coefficients, shape (num_timesteps,)
        sqrt_recipm1_alphas_cumprod (torch.Tensor): Precomputed coefficients, shape (num_timesteps,)
        betas_t (torch.Tensor): Beta values for each timestep, shape (num_timesteps,)

    Returns:
        torch.Tensor: Previous latent sequences, shape (B, T, C, H, W)
    """
    B, T, C, H, W = x_t.shape

    # Predict noise using the model
    eps_theta = model(x_t, t)  # Shape: (B, T, C, H, W)

    # Gather coefficients
    sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod[t].view(B, T, 1, 1, 1)
    sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod[t].view(B, T, 1, 1, 1)
    betas_t_current = betas_t[t].view(B, T, 1, 1, 1)

    # Compute the mean of x_{t-1}
    mean = sqrt_recip_alphas_cumprod_t * (x_t - betas_t_current * eps_theta / sqrt_recipm1_alphas_cumprod_t)

    # For simplicity, assume variance is beta_t
    var = betas_t_current

    # Sample noise
    noise = torch.randn_like(x_t)

    # Compute x_{t-1}
    x_prev = mean + torch.sqrt(var) * noise

    return x_prev