import torch
import torch.nn as nn
import torch.nn.functional as F
from schedules import get_beta_schedule


class DDPM(nn.Module):
    def __init__(self, model, timesteps=400, beta_schedule="cosine", beta_start=1e-4, beta_end=2e-2, device='cpu'):
        super().__init__()
        self.model = model
        self.device = device
        self.timesteps = timesteps
        betas = get_beta_schedule(beta_schedule, timesteps, beta_start, beta_end).to(device)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # precompute useful terms
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start, t, noise=None):
        """
        x_t = sqrt(alpha_cumprod[t]) * x_start + sqrt(1 - alpha_cumprod[t]) * noise
        t: tensor shape (b,)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        # gather factors
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_mean_variance(self, x_t, t):
        """
        Return model_mean, posterior_variance, x0_pred
        t: tensor shape (b,)
        """
        pred_noise = self.model(x_t, t)
        # predict x0
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_pred = sqrt_recip * x_t - sqrt_recipm1 * pred_noise
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # posterior mean coefficients
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        coef1 = betas_t * torch.sqrt(alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
        coef2 = (1.0 - alphas_cumprod_prev_t) * torch.sqrt(1.0 - betas_t) / (1.0 - alphas_cumprod_t)
        model_mean = coef1 * x0_pred + coef2 * x_t
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        return model_mean, posterior_variance, x0_pred

    def forward(self, x_start):
        """
        Compute training loss (MSE on predicted noise)
        x_start expected in [-1,1]
        """
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        pred_noise = self.model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, batch_size=16, channels=3, image_size=64, device=None):
        device = device or self.device
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        for t_int in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            model_mean, model_var, _ = self.p_mean_variance(x, t)
            if t_int > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(model_var) * noise
            else:
                x = model_mean
        return x
