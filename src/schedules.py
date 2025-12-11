import torch


def get_beta_schedule(schedule_name, timesteps, beta_start=0.0001, beta_end=0.02):
    if schedule_name == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name == "cosine":
        return cosine_beta_schedule(timesteps)
    elif schedule_name == "quadratic":
        return quadratic_beta_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Schedule inconnu: {schedule_name}")


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
