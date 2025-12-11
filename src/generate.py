"""
Script de génération d'images avec le modèle DDPM entraîné
"""

import os
import torch
from torchvision.utils import save_image
import argparse

from config import DiffusionConfig
from diffusion import DDPM, UNet
from dataset import denormalize


def load_model(checkpoint_path, device):
    """
    Charge un modèle depuis un checkpoint
    
    Args:
        checkpoint_path: Chemin vers le checkpoint
        device: Device (cuda/cpu)
    
    Returns:
        ddpm: Modèle chargé
        config: Configuration
    """
    print(f"📂 Chargement du modèle depuis {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Récupérer la config (ou utiliser la config par défaut si non disponible)
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = DiffusionConfig()
    
    # Créer le modèle U-Net
    unet = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        model_channels=config.model_channels,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        channel_mult=config.channel_mult,
        dropout=config.dropout,
        image_size=config.image_size
    ).to(device)
    
    # Créer le DDPM
    ddpm = DDPM(
        model=unet,
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        device=device
    ).to(device)
    
    # Charger les poids
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()
    
    print(f"✅ Modèle chargé avec succès!")
    
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"   Loss: {checkpoint['loss']:.4f}")
    
    return ddpm, config


def generate_images(
    checkpoint_path,
    num_images=16,
    output_dir='generated',
    device='cuda',
    save_grid=True,
    save_individual=False
):
    """
    Génère des images avec le modèle
    
    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        num_images: Nombre d'images à générer
        output_dir: Répertoire de sortie
        device: Device (cuda/cpu)
        save_grid: Sauvegarder en grille
        save_individual: Sauvegarder chaque image individuellement
    """
    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Charger le modèle
    ddpm, config = load_model(checkpoint_path, device)
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer les images
    print(f"\n🎨 Génération de {num_images} images...")
    print("⏳ Cela peut prendre plusieurs minutes...")
    
    with torch.no_grad():
        samples = ddpm.sample(
            batch_size=num_images,
            channels=config.in_channels,
            image_size=config.image_size
        )
    
    # Dénormaliser
    samples = denormalize(samples)
    
    # IMPORTANT: Clipper les valeurs pour éviter les artefacts
    samples = torch.clamp(samples, 0.0, 1.0)
    
    # Sauvegarder en grille
    if save_grid:
        grid_path = os.path.join(output_dir, 'generated_grid.png')
        save_image(samples, grid_path, nrow=4)
        print(f"✅ Grille sauvegardée: {grid_path}")
    
    # Sauvegarder individuellement
    if save_individual:
        for i, img in enumerate(samples):
            img_path = os.path.join(output_dir, f'generated_{i:04d}.png')
            save_image(img, img_path)
        print(f"✅ {num_images} images individuelles sauvegardées dans {output_dir}")
    
    print("\n✨ Génération terminée!")


def generate_with_steps(
    checkpoint_path,
    num_images=4,
    output_dir='generated_steps',
    device='cuda',
    save_every=100
):
    """
    Génère des images en sauvegardant les étapes intermédiaires
    Utile pour visualiser le processus de débruitage
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Charger le modèle
    ddpm, config = load_model(checkpoint_path, device)
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer avec étapes
    print(f"\n🎨 Génération de {num_images} images avec visualisation des étapes...")
    
    with torch.no_grad():
        steps = ddpm.sample_with_steps(
            batch_size=num_images,
            channels=config.in_channels,
            image_size=config.image_size,
            save_every=save_every
        )
    
    # Sauvegarder chaque étape
    for i, step_imgs in enumerate(steps):
        step_imgs = denormalize(step_imgs)
        # IMPORTANT: Clipper les valeurs
        step_imgs = torch.clamp(step_imgs, 0.0, 1.0)
        step_path = os.path.join(output_dir, f'step_{i:04d}.png')
        save_image(step_imgs, step_path, nrow=2)
    
    print(f"✅ {len(steps)} étapes sauvegardées dans {output_dir}")
    print("✨ Génération terminée!")


def main():
    parser = argparse.ArgumentParser(description='Génération d\'images avec DDPM')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--num_images', type=int, default=16,
                        help='Nombre d\'images à générer')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Répertoire de sortie')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save_individual', action='store_true',
                        help='Sauvegarder chaque image individuellement')
    parser.add_argument('--show_steps', action='store_true',
                        help='Visualiser les étapes de débruitage')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Sauvegarder une étape tous les N steps')
    
    args = parser.parse_args()
    
    if args.show_steps:
        generate_with_steps(
            checkpoint_path=args.checkpoint,
            num_images=min(4, args.num_images),
            output_dir=args.output_dir,
            device=args.device,
            save_every=args.save_every
        )
    else:
        generate_images(
            checkpoint_path=args.checkpoint,
            num_images=args.num_images,
            output_dir=args.output_dir,
            device=args.device,
            save_individual=args.save_individual
        )


if __name__ == '__main__':
    main()
