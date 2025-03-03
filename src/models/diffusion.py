import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os


def get_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.008):
    return torch.linspace(beta_start, beta_end, timesteps)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                        if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x))
        # Incorporate time embedding as a bias
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        h = F.relu(h)
        h = self.norm2(self.conv2(h))
        return F.relu(h + self.res_conv(x))

class SpatialTransformer(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x_reshape = x.view(B, C, H * W).transpose(1, 2)
        x_norm = self.norm(x_reshape)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        out = attn_output + x_reshape
        return out.transpose(1, 2).view(B, C, H, W)

class TemporalTransformer(nn.Module):
    def __init__(self, channels, num_time_steps, num_heads=4):
        super().__init__()
        self.num_time_steps = num_time_steps
        # Assume channels is divisible by num_time_steps.
        self.temporal_dim = channels // num_time_steps
        self.norm = nn.LayerNorm(self.temporal_dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.temporal_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x: [B, C, H, W] where C = num_time_steps * (channels_per_time)
        B, C, H, W = x.shape
        T = self.num_time_steps
        D = self.temporal_dim
        # Reshape into explicit time tokens: [B, T, D, H, W]
        x_reshaped = x.view(B, T, D, H, W)
        # For each spatial location, treat time as the sequence dimension.
        x_reshaped = x_reshaped.permute(0, 3, 4, 1, 2)  # [B, H, W, T, D]
        B, H, W, T, D = x_reshaped.shape
        x_seq = x_reshaped.contiguous().view(B * H * W, T, D)
        x_norm = self.norm(x_seq)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_output
        x_seq = x_seq.view(B, H, W, T, D).permute(0, 3, 4, 1, 2)  # Back to [B, T, D, H, W]
        out = x_seq.contiguous().view(B, T * D, H, W)
        return out

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, channels, num_time_steps, num_heads=4):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(channels, num_heads)
        self.temporal_transformer = TemporalTransformer(channels, num_time_steps, num_heads)
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self, x):
        residual = x
        x = self.spatial_transformer(x)
        x = self.temporal_transformer(x)
        # Apply MLP per spatial location
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        x_mlp = self.mlp(x_flat)
        x_mlp = x_mlp.transpose(1, 2).view(B, C, H, W)
        return x + residual + x_mlp

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, base_channels=64, time_emb_dim=128, transformer_time_steps=2):
        super().__init__()

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Down-sampling (encoder) blocks
        self.down1 = ResidualBlock(in_channels, base_channels, time_emb_dim)        # (45x90)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)    # (23x45)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # (12x23)
        self.down4 = ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim)  # (6x12)

        # Pooling for resolution reduction
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)  # Handles odd sizes

        # Cascaded Transformers at different levels
        self.transformer_low = DiffusionTransformerBlock(base_channels * 8, num_time_steps=transformer_time_steps, num_heads=4)   # 3x6 resolution
        self.transformer_mid = DiffusionTransformerBlock(base_channels * 4, num_time_steps=transformer_time_steps, num_heads=4)   # 6x12 resolution
        self.transformer_high = DiffusionTransformerBlock(base_channels * 2, num_time_steps=transformer_time_steps, num_heads=4)  # 12x23 resolution

        # Up-sampling (decoder) blocks
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim)  # (3x6) -> (6x12)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)  # (6x12) -> (12x23)
        self.up3 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)      # (12x23) -> (23x45)
        self.up4 = ResidualBlock(base_channels, base_channels, time_emb_dim)            # (23x45) -> (45x90)

        # Projection layers for skip connections to match channel dimensions
        self.proj_h4 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        self.proj_h3 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.proj_h2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def center_crop(self, x, target_shape):
        # x: tensor of shape [B, C, H, W]; target_shape: (target_H, target_W)
        _, _, h, w = x.shape
        target_h, target_w = target_shape
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return x[:, :, start_h:start_h+target_h, start_w:start_w+target_w]


    def forward(self, x, t):
        """
        x: Noisy latent tensor of shape [B, C, H, W].
        t: Time tensor of shape [B, 1], values scaled between 0 and 1.
        """
        t_emb = self.time_mlp(t)  # Compute time embedding

        # Encoder path
        h1 = self.down1(x, t_emb)       # (45x90)
        h = self.pool(h1)               # (23x45)
        h2 = self.down2(h, t_emb)       # (23x45)
        h = self.pool(h2)               # (12x23)
        h3 = self.down3(h, t_emb)       # (12x23)
        h = self.pool(h3)               # (6x12)
        h4 = self.down4(h, t_emb)       # (6x12)
        h = self.pool(h4)               # (3x6)

        # Transformer at the lowest level (long-term dependencies)
        h = self.transformer_low(h)

        # Decoder path
        h = self.upsample(h)  # From (3x6) to (6x12)
        up1 = self.up1(h, t_emb)
        proj_h4 = self.proj_h4(h4)
        if up1.shape[-2:] != proj_h4.shape[-2:]:
            up1 = self.center_crop(up1, proj_h4.shape[-2:])
        h = up1 + proj_h4
        h = self.transformer_mid(h)

        h = self.upsample(h)  # Expected (12x23) but might be (12x24)
        up2 = self.up2(h, t_emb)
        proj_h3 = self.proj_h3(h3)
        if up2.shape[-2:] != proj_h3.shape[-2:]:
            up2 = self.center_crop(up2, proj_h3.shape[-2:])
        h = up2 + proj_h3
        h = self.transformer_high(h)

        h = self.upsample(h)  # (23x45)
        up3 = self.up3(h, t_emb)
        proj_h2 = self.proj_h2(h2)
        if up3.shape[-2:] != proj_h2.shape[-2:]:
            up3 = self.center_crop(up3, proj_h2.shape[-2:])
        h = up3 + proj_h2

        h = self.upsample(h)  # (45x90)
        up4 = self.up4(h, t_emb)
        if up4.shape[-2:] != h1.shape[-2:]:
            up4 = self.center_crop(up4, h1.shape[-2:])
        h = up4 + h1

        h = self.final_conv(h)

        return h

class DiffusionModel:
    """
    A diffusion model that learns to predict the residual (or noise) in latent space.
    """
    def __init__(self, model, timesteps=1000, noise_scale=1.0, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.timesteps = timesteps
        self.betas = get_beta_schedule(timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.noise_scale = noise_scale

    def add_noise(self, x0, t, noise=None):
        """
        x0: Original residual (x0 - cond).
        t: A tensor of time steps (normalized between 0 and 1) of shape [B, 1].
        noise: Optional noise tensor; if None, new Gaussian noise is generated.
        Returns: (noisy residual, noise).
        """
        if noise is None:
            noise = torch.randn_like(x0) * self.noise_scale
        
        # Compute timestep indices for each sample in the batch
        t_idx = (t * self.timesteps).long().squeeze(1)  # shape [B]
        t_idx = t_idx.clamp(max=self.timesteps - 1)
        
        # Create a shape that will broadcast correctly
        coeff_shape = [x0.size(0)] + [1] * (x0.dim() - 1)
        
        # Apply noise according to schedule
        sqrt_alphas_cumprod = self.alphas_cumprod[t_idx].sqrt().view(*coeff_shape)
        sqrt_one_minus_alphas = (1 - self.alphas_cumprod[t_idx]).sqrt().view(*coeff_shape)
        
        noisy_x = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas * noise
        return noisy_x, noise

    def train_step(self, optimizer, x0, cond):
        """
        Performs one training step.
        x0: Clean latent from VAE.
        cond: Conditioning latent.
        """ 
        self.model.train()
        batch_size = x0.size(0)
        
        # Sample random timesteps uniformly and scale them to [0,1]
        t = torch.randint(0, self.timesteps, (batch_size, 1), device=self.device).float() / self.timesteps
        
        # Calculate residual
        residual = x0 - cond
        
        # Reshape for model
        residual = residual.view(batch_size, -1, *residual.shape[-2:])
        cond = cond.view(batch_size, -1, *cond.shape[-2:])
        
        # Add noise to the residual according to diffusion schedule
        _, noise = self.add_noise(residual, t)
        
        # Model predicts the noise
        pred_noise = self.model(cond, t)
        
        # Loss between actual noise and predicted noise
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(self, cond, shape, num_steps=None):
        """
        Modified diffusion sampling with controlled noise.
        """
        self.model.eval()
        batch_size = cond.size(0)
        device = self.device
        
        if num_steps is None:
            num_steps = min(100, self.timesteps)  # Use fewer steps by default
        else:
            num_steps = min(num_steps, self.timesteps)
        
        # Start with scaled-down noise - weather residuals have lower variance
        x = torch.randn(shape, device=device) * self.noise_scale
        cond = cond.view(batch_size, -1, *cond.shape[-2:])
        
        # Use evenly spaced steps
        step_indices = torch.linspace(0, self.timesteps-1, steps=num_steps, dtype=torch.long, device=device)
        
        # Perform reverse diffusion with gradual denoising
        for i in range(num_steps-1, -1, -1):
            index = step_indices[i]
            
            # Time embedding
            t = (index / self.timesteps) * torch.ones((batch_size, 1), device=device)
            
            # Get noise prediction
            predicted_noise = self.model(cond, t)
            
            # Calculate coefficients
            alpha = self.alphas[index]
            alpha_cumprod = self.alphas_cumprod[index]
            beta = self.betas[index]
            
            # For last few steps, reduce added noise for more coherent output
            if i > 0:
                # Progressively reduce noise as we approach final steps
                noise_factor = min(1.0, i / (num_steps * 0.75))
                noise = torch.randn_like(x) * noise_factor * self.noise_scale
                next_index = step_indices[i-1]
                next_alpha_cumprod = self.alphas_cumprod[next_index]
            else:
                noise = torch.zeros_like(x)
                next_alpha_cumprod = torch.ones_like(alpha_cumprod)
            
            # Standard equation for reverse diffusion step
            coefficient1 = 1 / alpha.sqrt()
            coefficient2 = (1 - alpha) / (1 - alpha_cumprod).sqrt()
            
            # Update x using predicted noise
            x = coefficient1 * (x - coefficient2 * predicted_noise)
            
            # Add controlled noise for the next step
            if i > 0:
                sigma = ((1 - next_alpha_cumprod) / (1 - alpha_cumprod) * beta).sqrt()
                x = x + sigma * noise
        
        # Optional: apply a light smoothing filter to the final residual
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        
        return x
    
    @torch.no_grad()
    def sample_ddim(self, cond, shape, num_steps=None):
        """
        Modified diffusion sampling with controlled noise.
        """
        self.model.eval()
        batch_size = cond.size(0)
        eta = 0.5  # 0 for deterministic sampling, 1.0 for stochastic

        if num_steps is None:
            num_steps = min(100, self.timesteps)  # Use fewer steps by default
        else:
            num_steps = min(num_steps, self.timesteps)
        step_indices = torch.round(torch.sqrt(torch.linspace(0, self.timesteps**2-1, 
                                     steps=num_steps, device=self.device))).clamp(max=self.timesteps-1).long()


        
        
        # Start with scaled-down noise - weather residuals have lower variance
        x = torch.randn(shape, device=self.device) * self.noise_scale
        cond = cond.view(batch_size, -1, *cond.shape[-2:])
        for i in range(num_steps-1, -1, -1):
            index = step_indices[i]
            alpha_cumprod = self.alphas_cumprod[index]
            
            t = (index / self.timesteps) * torch.ones((batch_size, 1), device=self.device)
            predicted_noise = self.model(cond, t)
            
            # Predict x0
            x0_pred = (x - torch.sqrt(1-alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            
            if i > 0:
                next_index = step_indices[i-1]
                next_alpha_cumprod = self.alphas_cumprod[next_index]
                
                # Calculate sigma for stochasticity control
                sigma = eta * torch.sqrt((1 - next_alpha_cumprod) / (1 - alpha_cumprod)) * torch.sqrt(1 - alpha_cumprod / next_alpha_cumprod)
                
                # DDIM update
                c1 = torch.sqrt(next_alpha_cumprod)
                c2 = torch.sqrt(1 - next_alpha_cumprod - sigma**2)
                noise = torch.randn_like(x) * self.noise_scale if sigma > 0 else 0
                x = c1 * x0_pred + c2 * predicted_noise + sigma * noise
            else:
                x = x0_pred  # Final step is deterministic

        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x