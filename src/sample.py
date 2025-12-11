import os
import torch
from utils import save_image_grid
from dataset import denormalize
from model import UNet
from diffusion import DDPM
from config import DiffusionConfig

def sample_from_checkpoint(ckpt_path, out_path='samples/sample_from_ckpt.png', num_samples=16, device=None):
    cfg = DiffusionConfig()
    device = device or (torch.device(cfg.device) if torch.cuda.is_available() else torch.device('cpu'))

    ckpt = torch.load(ckpt_path, map_location=device)
    unet = UNet(in_channels=cfg.in_channels, out_channels=cfg.out_channels, model_channels=cfg.model_channels, num_res_blocks=cfg.num_res_blocks, attention_resolutions=cfg.attention_resolutions, channel_mult=cfg.channel_mult, dropout=cfg.dropout, image_size=cfg.image_size).to(device)
    unet.load_state_dict(ckpt['unet_state_dict'])
    ddpm = DDPM(model=unet, timesteps=cfg.timesteps, beta_schedule=cfg.beta_schedule, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=device).to(device)

    # If EMA shadow present, copy weights into model
    if 'ema_shadow' in ckpt:
        shadow = ckpt['ema_shadow']
        for name, param in unet.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])

    ddpm.eval()
    with torch.no_grad():
        samples = ddpm.sample(batch_size=num_samples, channels=cfg.in_channels, image_size=cfg.image_size, device=device)
    samples = denormalize(samples.cpu())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image_grid(samples, out_path, nrow=4)
    print(f"Samples saved to {out_path}")

if __name__ == '__main__':
    # exemple d'utilisation
    sample_from_checkpoint('checkpoints/final_unet.pt', out_path='samples/from_final.png', num_samples=16)
