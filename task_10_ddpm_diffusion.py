"""
Task 10: Denoising Diffusion Probabilistic Model (DDPM) Forward & Reverse Latent Optimization
-----------------------------------------------------------------------------------------
Objective: Build deep generative models for image synthesis from first principles,
studying the transition from raw Gaussian noise to structured pixels.

Required Tech Stack: PyTorch, NumPy, Matplotlib
Formulas:
  Forward Process:  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
  Noise Schedule:   beta_t linear from beta_min to beta_max over T timesteps
  Reverse Process:  x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearVarianceScheduler:
    """
    Computes linear noise schedule parameters for DDPM.
    """
    def __init__(self, timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Precompute square roots for fast sampling
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward noise injection: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t


class LightweightDenoisingUNet(nn.Module):
    """
    Lightweight 2D Denoising Network predicting noise eps_theta(x_t, t).
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.mid_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Normalize timestep t to [0, 1]
        t_norm = (t.float() / 1000.0).view(-1, 1)
        t_emb = self.time_embed(t_norm).view(-1, 32, 1, 1)

        h = self.conv_in(x) + t_emb
        h = self.mid_block(h)
        noise_pred = self.conv_out(h)
        return noise_pred


class DDPMSampler:
    """
    Iterative Reverse Generative Sampling Loop p_theta(x_{t-1} | x_t).
    """
    def __init__(self, model: nn.Module, scheduler: LinearVarianceScheduler):
        self.model = model
        self.scheduler = scheduler

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t_idx: int) -> torch.Tensor:
        t_tensor = torch.tensor([t_idx], device=x_t.device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = self.model(x_t, t_tensor)

        beta_t = self.scheduler.beta[t_idx]
        sqrt_recip_alpha_t = self.scheduler.sqrt_recip_alpha[t_idx]
        sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alpha_bar[t_idx]

        # Mean calculation
        mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def generate_image(self, shape: tuple[int, int, int, int], steps: int = 1000) -> torch.Tensor:
        # Start from pure isotropic Gaussian noise x_T ~ N(0, I)
        x_t = torch.randn(shape)

        # Reverse loop from t = T-1 down to 0
        for t in range(steps - 1, -1, -1):
            x_t = self.p_sample(x_t, t)

        # Clamp pixels to [-1, 1]
        return torch.clamp(x_t, -1.0, 1.0)


def main():
    print("=" * 70)
    print("Task 10: Denoising Diffusion Probabilistic Model (DDPM) Verification")
    print("=" * 70)

    torch.manual_seed(42)
    timesteps = 1000
    resolution = (1, 1, 64, 64) # Single channel 64x64 image array

    # 1. Forward Noise Schedule Verification
    scheduler = LinearVarianceScheduler(timesteps=timesteps)
    x_0 = torch.zeros(resolution) # Clean target image (e.g. black canvas)

    t_mid = torch.tensor([500])
    x_500 = scheduler.q_sample(x_0, t_mid)
    
    print(f"Forward Process Verification:")
    print(f"  Target Image Shape:       {x_0.shape} (Resolution: 64x64)")
    print(f"  Timesteps T:             {timesteps}")
    print(f"  Alpha Bar at t=500:      {scheduler.alpha_bar[500]:.4f}")
    print(f"  Noised Image x_500 std:  {x_500.std().item():.4f}")

    # 2. Reverse Generative Loop Verification
    denoising_net = LightweightDenoisingUNet(in_channels=1, hidden_dim=32)
    sampler = DDPMSampler(denoising_net, scheduler)

    print("\n--- Running Reverse Generative Denoising Loop (64x64 over 1000 timesteps) ---")
    gen_image = sampler.generate_image(shape=resolution, steps=timesteps)

    print(f"Generative Denoising Complete:")
    print(f"  Generated Array Shape:   {gen_image.shape}")
    print(f"  Pixel Value Range:       [{gen_image.min().item():.3f}, {gen_image.max().item():.3f}]")

    print("\nTask 10 completed successfully!")

if __name__ == "__main__":
    main()
