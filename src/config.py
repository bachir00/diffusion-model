# class DiffusionConfig:
#     """Configuration du projet - ajuste ces valeurs selon ta VRAM"""

#     # Data
#     data_dir = "../data/train/cats"
#     image_size = 64
#     in_channels = 3
#     out_channels = 3

#     # Model (UNet)
#     model_channels = 192
#     channel_mult = [1, 2, 3, 4]
#     num_res_blocks = 2
#     attention_resolutions = [16, 8]
#     dropout = 0.1 # Changer selon la taille du dataset 

#     # Diffusion
#     timesteps = 1500
#     beta_schedule = "cosine"  # 'linear', 'cosine', 'quadratic'
#     beta_start = 0.0001  # utilisé seulement si linear/quadratic
#     beta_end = 0.02      # utilisé seulement si linear/quadratic

#     # Training
#     batch_size = 8
#     num_epochs = 600
#     learning_rate = 1e-4
#     num_workers = 4

#     # Checkpoint / sampling
#     checkpoint_dir = "checkpoints"
#     sample_dir = "samples"
#     num_samples = 16
#     sample_every_n_epochs = 5
#     save_checkpoint_every = 30

#     # EMA
#     ema_decay = 0.9999

#     # Misc
#     device = "cuda"


# # # Version 128x128
# class DiffusionConfig1:
#     """Optimisé pour RTX 4070 8 Go - 128x128 rapide"""

#     data_dir = "../data/train/cats_cleaned/good"
#     image_size = 128
#     in_channels = 3
#     out_channels = 3

#     # UNet léger mais performant
#     model_channels = 192          # 192 = très bon rapport qualité / VRAM
#     channel_mult = [1, 2, 3, 4]   # 128x128 parfait
#     num_res_blocks = 2
#     attention_resolutions = [16]  # 16 seulement = VRAM divisée par 2
#     dropout = 0.1

#     # Diffusion
#     timesteps = 1000               # rapide + bonne qualité
#     beta_schedule = "cosine"
#     beta_start = 0.0001
#     beta_end = 0.02

#     # Training
#     batch_size = 8                # tient dans 8 Go
#     num_epochs = 500              # suffisant pour 2000–3000 images
#     learning_rate = 2e-4
#     num_workers = 4

#     # Checkpoints & Samples
#     checkpoint_dir = "checkpoints128"
#     sample_dir = "samples128"
#     num_samples = 16
#     sample_every_n_epochs = 5
#     save_checkpoint_every = 50

#     # EMA
#     ema_decay = 0.999

#     # Misc
#     device = "cuda"

# class DiffusionConfig2:

#     # DATA
#     data_dir = "../data/train/cats_cleaned/good"
#     image_size = 128
#     in_channels = 3
#     out_channels = 3

#     # LIGHT UNet 128x128
#     model_channels = 96              # au lieu de 192
#     channel_mult = [1, 2, 4]        # 3 niveaux au lieu de 4
#     num_res_blocks = 2
#     attention_resolutions = [32]    # attention seulement à 32x32
#     dropout = 0.0                   # plus rapide + plus stable

#     # Diffusion
#     timesteps = 600
#     beta_schedule = "cosine"
#     beta_start = 0.0001
#     beta_end = 0.02

#     # Training
#     batch_size = 8                  # optimal pour 4070 8GB
#     num_epochs = 350
#     learning_rate = 2e-4
#     num_workers = 4

#     # Checkpoints
#     checkpoint_dir = "checkpoints128_alleged"
#     sample_dir = "samples128_alleged"
#     num_samples = 16
#     sample_every_n_epochs = 5
#     save_checkpoint_every = 50

#     # EMA
#     ema_decay = 0.999

#     device = "cuda"


class DiffusionConfig:
    """Configuration PRO pour RTX 4070 Laptop 8GB - Qualité Midjourney-like"""
    
    # DATA
    data_dir = "../data/train/cats_cleaned/good"
    image_size = 64                  # gardez 128 pour commencer
    in_channels = 3
    out_channels = 3

    # UNet Puissant - ~120M params
    model_channels = 160              # capacité élevée
    channel_mult = [1, 2, 3, 4]      # hiérarchie complète
    num_res_blocks = 3                # CRUCIAL pour qualité
    attention_resolutions = [8, 16, 32]  # multi-scale attention
    dropout = 0.1
    num_heads = 8                     # attention multi-têtes
    use_scale_shift_norm = True       # améliore stabilité
    
    # Diffusion - Configuration optimale
    timesteps = 1000
    beta_schedule = "cosine"          # meilleur pour images naturelles
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training OPTIMISÉ pour 8GB VRAM
    batch_size = 4                    # ⚠️ CRITIQUE : 4 au lieu de 6
    gradient_accumulation_steps = 4   # simule batch_size=16
    num_epochs = 300                  # 5000 images = besoin de + d'epochs
    learning_rate = 1e-4              # conservateur = stable
    warmup_steps = 1000               # stabilise début training
    num_workers = 4
    pin_memory = True                 # accélère transfert CPU->GPU
    
    # Mixed Precision (OBLIGATOIRE pour 8GB)
    use_fp16 = True                   # divise VRAM par ~2 !
    gradient_clip = 1.0               # évite explosions de gradient
    
    # EMA (crucial pour qualité finale)
    ema_decay = 0.9999                # très élevé = images plus propres
    
    # Checkpoints & Monitoring
    checkpoint_dir = "checkpoints128_pro"
    sample_dir = "samples128_pro"
    num_samples = 16
    sample_every_n_epochs = 30        # moins fréquent (gain de temps)
    save_checkpoint_every = 50
    resume_from_checkpoint = None     # chemin si vous reprenez training
    
    # Optimisations VRAM
    empty_cache_every_n_steps = 50    # nettoie VRAM régulièrement
    cudnn_benchmark = True            # accélère convolutions
    
    device = "cuda"
