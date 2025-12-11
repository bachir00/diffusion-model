# import os
# import logging
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from config import DiffusionConfig
# from model import UNet
# from diffusion import DDPM
# from dataset import CatDataset, denormalize
# from util import EMA, save_image_grid

# # Setup logging with UTF-8 encoding
# import sys

# # Configure file handler with UTF-8 encoding
# file_handler = logging.FileHandler('training.log', encoding='utf-8')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# # Configure console handler with UTF-8 encoding
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# # Setup logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# def create_directories(config):
#     os.makedirs(config.checkpoint_dir, exist_ok=True)
#     os.makedirs(config.sample_dir, exist_ok=True)

# def save_samples(ddpm, unet, ema, config, epoch, device):
#     logger.info(f"[SAMPLING] Génération d'échantillons pour l'epoch {epoch}...")
#     ddpm.eval()
#     if ema is not None:
#         ema.apply_shadow()
#     with torch.no_grad():
#         samples = ddpm.sample(batch_size=config.num_samples, channels=config.in_channels, image_size=config.image_size, device=device)
#     samples = denormalize(samples.cpu())
#     save_image_grid(samples, os.path.join(config.sample_dir, f'samples_epoch_{epoch:04d}.png'), nrow=4)
#     logger.info(f"[SUCCESS] Échantillons sauvegardés: {os.path.join(config.sample_dir, f'samples_epoch_{epoch:04d}.png')}")
#     if ema is not None:
#         ema.restore()
#     ddpm.train()

# def train(resume_from_checkpoint=None):
#     config = DiffusionConfig()
#     device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
#     logger.info(f"[DEVICE] {device}")
#     create_directories(config)

#     dataset = CatDataset(config.data_dir, image_size=config.image_size, augment=True)
#     dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True if device.type=='cuda' else False)
#     logger.info(f"[DATASET] Loaded {len(dataset)} images")

#     unet = UNet(in_channels=config.in_channels, out_channels=config.out_channels, model_channels=config.model_channels, num_res_blocks=config.num_res_blocks, attention_resolutions=config.attention_resolutions, channel_mult=config.channel_mult, dropout=config.dropout, image_size=config.image_size).to(device)
#     num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
#     logger.info(f"[MODEL] UNet parameters: {num_params:,}")

#     ddpm = DDPM(model=unet, timesteps=config.timesteps, beta_schedule=config.beta_schedule, beta_start=config.beta_start, beta_end=config.beta_end, device=device).to(device)

#     # optimizer on UNet only
#     optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)

#     # warmup helper
#     def get_lr(step, warmup_steps=500):
#         if step < warmup_steps:
#             return float(step) / float(max(1, warmup_steps))
#         return 1.0

#     ema = EMA(unet, decay=config.ema_decay)

#     start_epoch = 0
#     global_step = 0
#     losses_epoch = []

#     if resume_from_checkpoint:
#         ckpt = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
#         unet.load_state_dict(ckpt['unet_state_dict'])
#         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#         start_epoch = ckpt.get('epoch', 0)
#         if 'ema_shadow' in ckpt:
#             ema.shadow = ckpt['ema_shadow']
#         logger.info(f"[RESUME] Loaded checkpoint from epoch {start_epoch}: {resume_from_checkpoint}")

#     for epoch in range(start_epoch, config.num_epochs):
#         unet.train()
#         epoch_loss = 0.0
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
#         for batch_idx, images in enumerate(pbar):
#             images = images.to(device)
#             loss = ddpm(images)
#             optimizer.zero_grad()
#             loss.backward()
#             # clip gradients on UNet
#             torch.nn.utils.clip_grad_norm_(unet.parameters(), 5.0)
#             # lr warmup
#             lr_mult = get_lr(global_step, warmup_steps=500)
#             for g in optimizer.param_groups:
#                 g['lr'] = config.learning_rate * lr_mult
#             optimizer.step()

#             # update EMA
#             ema.update()

#             batch_loss = loss.item()
#             epoch_loss += batch_loss
#             global_step += 1
#             pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg_loss': f'{(epoch_loss/(batch_idx+1)):.4f}'})

#         avg_epoch_loss = epoch_loss / len(dataloader)
#         logger.info(f"[EPOCH {epoch+1}/{config.num_epochs}] Average loss: {avg_epoch_loss:.4f}")

#         # save epoch loss history
#         losses_epoch.append(avg_epoch_loss)

#         # sampling
#         if (epoch + 1) % config.sample_every_n_epochs == 0:
#             save_samples(ddpm, unet, ema, config, epoch+1, device)

#         # checkpoint
#         if (epoch + 1) % config.save_checkpoint_every == 0:
#             ckpt = {
#                 'epoch': epoch + 1,
#                 'unet_state_dict': unet.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_epoch_loss,
#                 'ema_shadow': ema.shadow,
#                 'config': config
#             }
#             path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pt')
#             torch.save(ckpt, path)
#             logger.info(f"[CHECKPOINT] Saved: {path}")

#     # final save
#     final_path = os.path.join(config.checkpoint_dir, 'final_unet.pt')
#     torch.save({'unet_state_dict': unet.state_dict(), 'ema_shadow': ema.shadow, 'config': config}, final_path)
#     logger.info(f"[COMPLETE] Final model saved: {final_path}")

#     # final samples
#     save_samples(ddpm, unet, ema, config, 'final', device)

# if __name__ == '__main__':
#     train()

import os
import logging
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # ✅ CORRECTION: Nouveau import
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time

from config import DiffusionConfig
from model import UNet
from diffusion import DDPM
from dataset import CatDataset, denormalize
from util import EMA, save_image_grid

# Setup logging with UTF-8 encoding
import sys

# Configure file handler with UTF-8 encoding
file_handler = logging.FileHandler('training.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure console handler with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def create_directories(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

def save_samples(ddpm, unet, ema, config, epoch, device):
    logger.info(f"[SAMPLING] Génération d'échantillons pour l'epoch {epoch}...")
    ddpm.eval()
    if ema is not None:
        ema.apply_shadow()
    with torch.no_grad():
        # Use FP32 for sampling to ensure quality
        with autocast(device_type='cuda', enabled=False):  # ✅ CORRECTION
            samples = ddpm.sample(batch_size=config.num_samples, channels=config.in_channels, 
                                image_size=config.image_size, device=device)
    samples = denormalize(samples.cpu())
    save_image_grid(samples, os.path.join(config.sample_dir, f'samples_epoch_{epoch:04d}.png'), nrow=4)
    logger.info(f"[SUCCESS] Échantillons sauvegardés: {os.path.join(config.sample_dir, f'samples_epoch_{epoch:04d}.png')}")
    if ema is not None:
        ema.restore()
    ddpm.train()

def setup_training_optimizations(config):
    """Configure CUDA optimizations for maximum performance"""
    if hasattr(config, 'cudnn_benchmark') and config.cudnn_benchmark:
        cudnn.benchmark = True
        logger.info("[OPTIMIZATION] cuDNN benchmark enabled")
    
    # Enable TF32 for Ampere GPUs (RTX 30/40 series)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("[OPTIMIZATION] TF32 enabled for Ampere GPU")

def train(resume_from_checkpoint=None):
    config = DiffusionConfig()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"[DEVICE] {device}")
    
    # Apply CUDA optimizations
    setup_training_optimizations(config)
    
    create_directories(config)

    # Dataset with augmentation
    dataset = CatDataset(config.data_dir, image_size=config.image_size, augment=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=getattr(config, 'pin_memory', True) if device.type == 'cuda' else False,
        persistent_workers=True if config.num_workers > 0 else False  # Faster dataloader
    )
    logger.info(f"[DATASET] Loaded {len(dataset)} images")

    # Build UNet model
    unet_kwargs = {
        'in_channels': config.in_channels,
        'out_channels': config.out_channels,
        'model_channels': config.model_channels,
        'num_res_blocks': config.num_res_blocks,
        'attention_resolutions': config.attention_resolutions,
        'channel_mult': config.channel_mult,
        'dropout': config.dropout,
        'image_size': config.image_size
    }
    
    # Add optional parameters if they exist
    if hasattr(config, 'num_heads'):
        unet_kwargs['num_heads'] = config.num_heads
    if hasattr(config, 'use_scale_shift_norm'):
        unet_kwargs['use_scale_shift_norm'] = config.use_scale_shift_norm
    
    unet = UNet(**unet_kwargs).to(device)
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"[MODEL] UNet parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    ddpm = DDPM(
        model=unet, 
        timesteps=config.timesteps, 
        beta_schedule=config.beta_schedule, 
        beta_start=config.beta_start, 
        beta_end=config.beta_end, 
        device=device
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # Learning rate scheduler with warmup
    warmup_steps = getattr(config, 'warmup_steps', 1000)
    def get_lr_multiplier(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    # Mixed Precision Scaler - ✅ CORRECTION: Nouveau format
    use_fp16 = getattr(config, 'use_fp16', False)
    scaler = GradScaler(device='cuda', enabled=use_fp16)
    if use_fp16:
        logger.info("[OPTIMIZATION] Mixed Precision (FP16) enabled - VRAM usage reduced by ~40%")

    # Gradient accumulation
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    logger.info(f"[TRAINING] Batch size: {config.batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    # effective_batch_size = config.batch_size * gradient_accumulation_steps
    # logger.info(f"[TRAINING] Effective batch size: {effective_batch_size}")

    # Gradient clipping value
    gradient_clip = getattr(config, 'gradient_clip', 1.0)

    # EMA
    ema = EMA(unet, decay=config.ema_decay)
    logger.info(f"[EMA] Decay: {config.ema_decay}")

    start_epoch = 0
    global_step = 0
    losses_epoch = []
    best_loss = float('inf')

    # Resume from checkpoint
    if resume_from_checkpoint:
        ckpt = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt['unet_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        if 'ema_shadow' in ckpt:
            ema.shadow = ckpt['ema_shadow']
        if 'scaler_state_dict' in ckpt and use_fp16:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        logger.info(f"[RESUME] Loaded checkpoint from epoch {start_epoch}: {resume_from_checkpoint}")

    # VRAM monitoring
    empty_cache_every = getattr(config, 'empty_cache_every_n_steps', 50)

    logger.info("="*80)
    logger.info("[START] Training begins!")
    logger.info("="*80)

    for epoch in range(start_epoch, config.num_epochs):
        unet.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            
            # Mixed precision forward pass - ✅ CORRECTION: device_type au lieu de enabled
            with autocast(device_type='cuda', enabled=use_fp16):
                loss = ddpm(images)
                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation: only update weights every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(unet.parameters(), gradient_clip)
                
                # Learning rate warmup
                lr_mult = get_lr_multiplier(global_step)
                for g in optimizer.param_groups:
                    g['lr'] = config.learning_rate * lr_mult
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update EMA after weight update
                ema.update()
                
                global_step += 1
            
            # Periodic VRAM cleanup
            if batch_idx % empty_cache_every == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
            
            # Logging
            batch_loss = loss.item() * gradient_accumulation_steps  # Rescale for display
            epoch_loss += batch_loss
            
            # Calculate current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # VRAM usage
            if device.type == 'cuda':
                vram_allocated = torch.cuda.memory_allocated() / 1024**3
                vram_reserved = torch.cuda.memory_reserved() / 1024**3
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg': f'{(epoch_loss/(batch_idx+1)):.4f}',
                    'lr': f'{current_lr:.2e}',
                    'vram': f'{vram_allocated:.2f}GB'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg': f'{(epoch_loss/(batch_idx+1)):.4f}',
                    'lr': f'{current_lr:.2e}'
                })

        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"[EPOCH {epoch+1}/{config.num_epochs}] Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s")
        
        if device.type == 'cuda':
            max_vram = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"[VRAM] Peak usage: {max_vram:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # Save epoch loss history
        losses_epoch.append(avg_epoch_loss)

        # Track best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'ema_shadow': ema.shadow,
                'global_step': global_step,
                'config': config
            }, best_path)
            logger.info(f"[BEST MODEL] Saved with loss: {best_loss:.4f}")

        # Sampling
        if (epoch + 1) % config.sample_every_n_epochs == 0:
            save_samples(ddpm, unet, ema, config, epoch+1, device)

        # Checkpoint
        if (epoch + 1) % config.save_checkpoint_every == 0:
            ckpt = {
                'epoch': epoch + 1,
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'ema_shadow': ema.shadow,
                'global_step': global_step,
                'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
                'config': config
            }
            path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pt')
            torch.save(ckpt, path)
            logger.info(f"[CHECKPOINT] Saved: {path}")

    # Final save
    final_path = os.path.join(config.checkpoint_dir, 'final_unet.pt')
    torch.save({
        'unet_state_dict': unet.state_dict(), 
        'ema_shadow': ema.shadow, 
        'config': config,
        'final_loss': avg_epoch_loss,
        'losses_history': losses_epoch
    }, final_path)
    logger.info(f"[COMPLETE] Final model saved: {final_path}")
    logger.info(f"[COMPLETE] Best loss achieved: {best_loss:.4f}")

    # Final samples with EMA
    save_samples(ddpm, unet, ema, config, 'final', device)
    
    logger.info("="*80)
    logger.info("[FINISHED] Training complete!")
    logger.info("="*80)

if __name__ == '__main__':
    # Pour reprendre l'entraînement, décommentez et spécifiez le chemin:
    train(resume_from_checkpoint='checkpoints128_pro/checkpoint_epoch_0233.pt')
    # train()