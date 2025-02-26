import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

####################################################
# Spatial & Temporal Attention Blocks
####################################################
class SpatialAttention3D(nn.Module):
    """
    Apply spatial attention per timestep (H, W) while preserving T.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Simple Transformer MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        We'll treat each T slice independently for spatial attention:
        flatten (H*W), keep T dimension in the batch iteration.
        """
        B, T, C, H, W = x.shape
        x_out = []

        for t in range(T):
            # (B, C, H, W) => (B, N, C), N = H*W
            x_t = x[:, t]
            x_t = x_t.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

            # === Pre-Attention LN ===
            x_t_norm = self.norm1(x_t)                   # (B, N, C)

            # === Multi-Head Attention ===
            qkv = self.qkv(x_t_norm)                     # (B, N, 3*C)
            q, k, v = torch.chunk(qkv, 3, dim=2)

            head_dim = C // self.num_heads
            q = q.view(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)

            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn = torch.softmax(scores, dim=-1)
            out_attn = torch.matmul(attn, v)  # (B, heads, N, head_dim)
            out_attn = out_attn.permute(0, 2, 1, 3).reshape(B, -1, C)
            out_attn = self.proj_out(out_attn)  # (B, N, C)

            # === Residual Connection for Attention ===
            x_t = x_t + out_attn

            # === Post-Attention LN + MLP ===
            x_t_norm2 = self.norm2(x_t)
            out_mlp = self.mlp(x_t_norm2)

            # === MLP Residual Connection ===
            x_t = x_t + out_mlp

            # Reshape back to (B, C, H, W)
            x_t = x_t.permute(0, 2, 1).reshape(B, C, H, W)
            x_out.append(x_t)

        # Stack along T => (B, T, C, H, W)
        return torch.stack(x_out, dim=1)

class TemporalAttention3D(nn.Module):
    def __init__(self, dim, num_heads, patch_factor=1):
        """
        Args:
            dim: The per-timestep feature dimension (C*H*W if flattened).
            num_heads: Number of attention heads.
            patch_factor: Controls how many chunks to split T into.
                            e.g., patch_factor=2 => 2 patches; patch_factor=4 => 4 patches, etc.
        """
        super().__init__()
        self.num_heads = num_heads
        self.patch_factor = patch_factor
        self.lower_dim = nn.Linear(1036800, 256)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        We'll split T into patches, flatten (C*H*W) as the feature dimension in each patch, and attend over that chunk.
        """
        B, T, C, H, W = x.shape
        feature_dim = C * H * W

        # Decide the patch size based on T and patch_factor.
        # e.g., patch_factor=2 means 2 patches (T/2 each) if T is divisible by 2.
        patch_size = max(1, T // self.patch_factor)

        outputs = []
        # === Process time in patches ===
        for start in range(0, T, patch_size):
            end = min(start + patch_size, T)
            chunk_length = end - start

            # Extract chunk => (B, chunk_length, C, H, W)
            x_chunk = x[:, start:end]  # shape: (B, chunk_length, C, H, W)
            x_chunk = x_chunk.view(B, chunk_length, feature_dim)  # flatten spatial dims
            x_chunk = self.lower_dim(x_chunk)
            # === Pre-Attention LN ===
            x_norm = self.norm1(x_chunk)  # shape: (B, chunk_length, feature_dim)

            # === Multi-Head Attention ===
            qkv = self.qkv(x_norm)  # (B, chunk_length, 3 * feature_dim)
            q, k, v = torch.chunk(qkv, chunks=3, dim=2)

            head_dim = feature_dim // self.num_heads
            q = q.view(B, chunk_length, self.num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(B, chunk_length, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(B, chunk_length, self.num_heads, head_dim).permute(0, 2, 1, 3)

            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn = torch.softmax(scores, dim=-1)
            out_attn = torch.matmul(attn, v)  # (B, heads, chunk_length, head_dim)

            # Reshape -> (B, chunk_length, feature_dim)
            out_attn = out_attn.permute(0, 2, 1, 3).reshape(B, chunk_length, feature_dim)
            out_attn = self.proj_out(out_attn)

            # === Residual Connection for Attention ===
            x_chunk = x_chunk + out_attn

            # === Post-Attention LN + MLP ===
            x_norm2 = self.norm2(x_chunk)
            out_mlp = self.mlp(x_norm2)

            # === MLP Residual Connection ===
            x_chunk = x_chunk + out_mlp

            # Reshape to original (B, chunk_length, C, H, W)
            x_chunk = x_chunk.view(B, chunk_length, C, H, W)
            outputs.append(x_chunk)

        # === Concatenate all patches along T ===
        return torch.cat(outputs, dim=1)

####################################################
# UNet Block + Transformer
####################################################
class ResidualBlock3D(nn.Module):
    """
    A 3D residual block that treats time as an explicit dimension
    without flattening (B, T, C, H, W).
    Convolutions are applied over (C, H, W) but keep T intact.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # We use Conv3d with kernel_size=(1,3,3) to preserve the T dimension
        # as is, but still convolve over H and W.
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.norm1 = nn.GroupNorm(4, out_c)  # group norm over out_c channels
        self.norm2 = nn.GroupNorm(4, out_c)
        self.activation = nn.SiLU()
        if in_c != out_c:
            self.residual_conv = nn.Conv3d(in_c, out_c, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        We permute to (B, C, T, H, W) for Conv3d, then permute back.
        """
        B, T, C, H, W = x.shape
        x_3d = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        residual = x_3d

        x_3d = self.conv1(x_3d)
        x_3d = self.norm1(x_3d)
        x_3d = self.activation(x_3d)

        x_3d = self.conv2(x_3d)
        x_3d = self.norm2(x_3d)
        if self.residual_conv:
            residual = self.residual_conv(residual)

        x_3d = residual + self.activation(x_3d)
        # Permute back to (B, T, C, H, W)
        x_out = x_3d.permute(0, 2, 1, 3, 4)
        return x_out

class DiffusionUNet(nn.Module):
    """
    A U-Net style architecture that uses:
    - 3D Residual Blocks (no flattening)
    - SpatialAttention3D
    - TemporalAttention3D
    """
    def __init__(self, c_in=4, c_out=4, time_embed_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Example down path using 3D blocks
        self.down1 = ResidualBlock3D(c_in, 64)
        self.down2 = ResidualBlock3D(64, 128)
        self.down3 = ResidualBlock3D(128, 256)

        # Spatial + Temporal attention in the bottleneck
        self.spatial_attn = SpatialAttention3D(256, num_heads=4)
        self.temporal_attn = TemporalAttention3D(256, num_heads=2)

        # Example up path
        self.up1 = ResidualBlock3D(256, 128)
        self.up2 = ResidualBlock3D(128, 64)
        self.up3 = ResidualBlock3D(64, c_out)

    def forward(self, x, t):
        """
        x: (B, T, c_in, H, W)
        t: (B, T) if you sample separate timesteps per (B,T).
           If a single timestep per batch, t might be shape (B,).
        """
        # Down
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Attention in the bottleneck
        x3 = self.spatial_attn(x3)
        x3 = self.temporal_attn(x3)

        # Up
        # Simple skip-connections: just add x2, x1, etc.
        x4 = self.up1(x3)
        x5 = self.up2(x4 + x2)
        out = self.up3(x5 + x1)
        return out

    def _time_encoding(self, t, device):
        """
        Simplified sinusoidal positional encoding for t.
        """
        half_dim = 128
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        theta = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # shape can vary with T
        emb = torch.cat((theta.sin(), theta.cos()), dim=-1)   # (B, T, 2*half_dim)
        return emb

####################################################
# Main Diffusion Model: Handling Latent Space + Loss
####################################################
class DiffusionModel(nn.Module):
    def __init__(self, unet, num_timesteps=1000):
        super().__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps
        # Precompute betas and alphas
        self.betas = utils.make_beta_schedule(num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = 1.0 / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def forward(self, x_noisy, t):
        """
        Predict the noise given x_t.
        """
        return self.unet(x_noisy, t)

    def training_loss(self, vae, x0):
        """
        Compute the training loss for a batch of sequences.

        Args:
            vae (nn.Module): VAE model for encoding/decoding.
            x0 (torch.Tensor): Original image sequences, shape (B, T, 3, H, W)

        Returns:
            torch.Tensor: MSE loss between predicted noise and actual noise.
        """
        # 1) Encode the original images into latent space
        with torch.no_grad():
            latent_seq = utils.encode_sequence_with_vae(vae, x0)  # Shape: (B, T, C, H, W)

        B, T, C, H, W = latent_seq.shape

        # 2) Sample random timesteps for each element in the batch and each timestep
        t = torch.randint(0, self.num_timesteps, (B, T), device=latent_seq.device).long()  # Shape: (B, T)

        # 3) Apply the forward diffusion process to get x_t and noise

        x_t, noise = utils.q_sample(latent_seq, t, 
                                    self.sqrt_alphas_cumprod, 
                                    self.sqrt_one_minus_alphas_cumprod)

        # 5) Predict the noise using the U-Net model
        noise_pred = self.forward(x_t, t)  # Shape: (B, T, C, H, W)

        # 6) Compute the Mean Squared Error loss between predicted noise and actual noise
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def sample(self, vae, shape, device='cuda'):
        """
        Reverse diffusion process to generate samples from noise.

        Args:
            vae (nn.Module): VAE model for decoding latent space.
            shape (tuple): Shape of the latent tensor to sample, (B, T, C, H, W)
            device (str): Device to perform computations on.

        Returns:
            torch.Tensor: Generated image sequences, shape (B, T, 3, H, W)
        """
        B, T, C, H, W = shape
        x = torch.randn(shape, device=device)  # Initialize with noise

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B, T), i, device=device, dtype=torch.long)  # Shape: (B, T)
            x = utils.p_sample(self, x, t, 
                        self.sqrt_recip_alphas_cumprod, 
                        self.sqrt_recipm1_alphas_cumprod,
                        self.betas)
        
        with torch.no_grad():
            # Decode the latent samples back to image space
            decoded = vae.decode_sequence_with_vae(x)  # Shape: (B, T, 3, H, W)
        return decoded

####################################################
# Pseudo-Training Loop
####################################################
def train_diffusion(diffusion_model, vae, data_loader, optimizer, device='cuda'):
    diffusion_model.train()
    diffusion_model.to(device)
    for epoch in range(10):  # placeholder
        for batch_idx, real_images in enumerate(data_loader):
            real_images = real_images.to(device)
            optimizer.zero_grad()
            loss = diffusion_model.training_loss(vae, real_images)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

####################################################
# How to Use (Simplified Example)
####################################################

import data_utils as du
from torch.utils.data import DataLoader
import variational_encoder as ve

if __name__ == "__main__":
    # Charger le dataset
    dataset = du.open_grib_file("data.grib")

    # Définir les paramètres de fenêtrage
    window_size = 2  # Nombre d'étapes temporelles par fenêtre
    overlap = 1       # Chevauchement entre fenêtres

    # Initialiser le dataset
    image_sequence_dataset = du.ImageSequenceDataset(
        dataset=dataset,
        window_size=window_size,
        overlap=overlap,
        variables=['t2m', 'sp', 'tcc'],
        skip_last=True
    )
    
    # Initialiser le DataLoader
    data_loader = DataLoader(
        image_sequence_dataset,
        batch_size=2,           # Nombre de fenêtres par batch
        shuffle=True,           # Mélanger les données
        num_workers=0,          # Nombre de workers pour le chargement des données
        drop_last=True          # Ignorer les batches incomplets
    )

    vae = ve.VAE()
    unet = DiffusionUNet(c_in=4, c_out=4)
    diffusion_model = DiffusionModel(unet=unet, num_timesteps=1000)

    # Example: define optimizer
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4)

    train_diffusion(diffusion_model, vae, data_loader, optimizer, device='cuda')

    # Sampling
    samples = diffusion_model.sample(vae, (2,4,64,64), device='cuda')
    print("Sampled shape:", samples.shape)