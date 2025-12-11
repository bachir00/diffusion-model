# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SiLU(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


# def timestep_embedding(timesteps, dim, max_period=10000):
#     """
#     timesteps: tensor shape (b,)
#     dim: embedding dim
#     """
#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
#     ).to(timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         emb = F.pad(emb, (0, 1))
#     return emb


# class ResidualBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, temb_dim=None, dropout=0.0):
#         super().__init__()
#         self.norm1 = nn.GroupNorm(8, in_ch)
#         self.act = SiLU()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
#         self.norm2 = nn.GroupNorm(8, out_ch)
#         self.dropout = nn.Dropout(dropout)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
#         self.nin_shortcut = None
#         if in_ch != out_ch:
#             self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1)

#         if temb_dim is not None:
#             self.temb_proj = nn.Linear(temb_dim, out_ch)
#         else:
#             self.temb_proj = None

#     def forward(self, x, temb=None):
#         h = self.norm1(x)
#         h = self.act(h)
#         h = self.conv1(h)
#         if self.temb_proj is not None and temb is not None:
#             h = h + self.temb_proj(self.act(temb))[:, :, None, None]
#         h = self.norm2(h)
#         h = self.act(h)
#         h = self.dropout(h)
#         h = self.conv2(h)
#         if self.nin_shortcut is not None:
#             x = self.nin_shortcut(x)
#         return x + h


# class AttentionBlock(nn.Module):
#     def __init__(self, ch):
#         super().__init__()
#         self.norm = nn.GroupNorm(8, ch)
#         self.q = nn.Conv2d(ch, ch, 1)
#         self.k = nn.Conv2d(ch, ch, 1)
#         self.v = nn.Conv2d(ch, ch, 1)
#         self.proj = nn.Conv2d(ch, ch, 1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_norm = self.norm(x)
#         q = self.q(x_norm).reshape(b, c, -1).permute(0, 2, 1)  # b,hw,c
#         k = self.k(x_norm).reshape(b, c, -1)  # b,c,hw
#         v = self.v(x_norm).reshape(b, c, -1).permute(0, 2, 1)
#         attn = torch.bmm(q, k) * (c ** (-0.5))
#         attn = torch.softmax(attn, dim=-1)
#         out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
#         out = self.proj(out)
#         return x + out


# class UNet(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         out_channels=3,
#         model_channels=128,
#         channel_mult=(1, 2, 4, 8),
#         num_res_blocks=2,
#         attention_resolutions=(16, 8),
#         dropout=0.0,
#         image_size=64,
#     ):
#         super().__init__()
#         self.model_channels = model_channels
#         self.num_res_blocks = num_res_blocks
#         self.channel_mult = channel_mult
#         self.image_size = image_size

#         # time embedding
#         time_dim = model_channels * 4
#         self.time_mlp = nn.Sequential(
#             nn.Linear(model_channels, time_dim),
#             SiLU(),
#             nn.Linear(time_dim, time_dim),
#         )
#         self.time_dim = time_dim

#         # input conv
#         self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

#         # down sampling blocks
#         self.down = nn.ModuleList()
#         ch = model_channels
#         for i, mult in enumerate(channel_mult):
#             out_ch = model_channels * mult
#             for _ in range(num_res_blocks):
#                 self.down.append(ResidualBlock(ch, out_ch, temb_dim=time_dim, dropout=dropout))
#                 ch = out_ch
#             if (image_size // (2 ** i)) in attention_resolutions:
#                 self.down.append(AttentionBlock(ch))
#             if i != len(channel_mult) - 1:
#                 self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))  # downsample

#         # middle
#         self.mid1 = ResidualBlock(ch, ch, temb_dim=time_dim, dropout=dropout)
#         self.mid_attn = AttentionBlock(ch)
#         self.mid2 = ResidualBlock(ch, ch, temb_dim=time_dim, dropout=dropout)

#         # up sampling blocks
#         self.up = nn.ModuleList()
#         for i, mult in reversed(list(enumerate(channel_mult))):
#             out_ch = model_channels * mult
#             # Upsample FIRST (except for the first iteration which is at the lowest resolution)
#             if i != len(channel_mult) - 1:
#                 self.up.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
#             for _ in range(num_res_blocks):
#                 self.up.append(ResidualBlock(ch + out_ch, out_ch, temb_dim=time_dim, dropout=dropout))
#                 ch = out_ch
#             if (image_size // (2 ** i)) in attention_resolutions:
#                 self.up.append(AttentionBlock(ch))

#         # output
#         self.norm_out = nn.GroupNorm(8, ch)
#         self.act = SiLU()
#         self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

#     def forward(self, x, timesteps):
#         """
#         x: [b, c, h, w] values in [-1,1]
#         timesteps: tensor shape (b,) dtype long
#         """
#         temb = timestep_embedding(timesteps, self.model_channels).to(x.device)
#         temb = self.time_mlp(temb)

#         hs = []
#         h = self.conv_in(x)
#         hs.append(h)

#         # down
#         for layer in self.down:
#             if isinstance(layer, ResidualBlock):
#                 h = layer(h, temb)
#                 hs.append(h)
#             elif isinstance(layer, AttentionBlock):
#                 h = layer(h)
#                 # Do NOT append to hs after attention
#             else:  # conv downsample
#                 h = layer(h)
#                 # Do NOT append here - upsample doesn't need this skip

#         # middle
#         h = self.mid1(h, temb)
#         h = self.mid_attn(h)
#         h = self.mid2(h, temb)

#         # up
#         for layer in self.up:
#             if isinstance(layer, nn.ConvTranspose2d):  # upsample FIRST
#                 h = layer(h)
#             elif isinstance(layer, ResidualBlock):
#                 skip = hs.pop()
#                 h = torch.cat([h, skip], dim=1)
#                 h = layer(h, temb)
#             elif isinstance(layer, AttentionBlock):
#                 h = layer(h)

#         h = self.norm_out(h)
#         h = self.act(h)
#         return self.conv_out(h)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    timesteps: tensor shape (b,)
    dim: embedding dim
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim=None, dropout=0.0, use_scale_shift_norm=False):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.nin_shortcut = None
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1)

        if temb_dim is not None:
            # Scale-shift norm projette 2x plus de channels (scale + shift)
            out_dim = 2 * out_ch if use_scale_shift_norm else out_ch
            self.temb_proj = nn.Linear(temb_dim, out_dim)
        else:
            self.temb_proj = None

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        if self.temb_proj is not None and temb is not None:
            temb_out = self.temb_proj(self.act(temb))[:, :, None, None]
            if self.use_scale_shift_norm:
                # Split en scale et shift
                scale, shift = torch.chunk(temb_out, 2, dim=1)
                h = self.norm2(h) * (1 + scale) + shift
            else:
                h = h + temb_out
                h = self.norm2(h)
        else:
            h = self.norm2(h)
        
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head self-attention (version originale)"""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, c, -1).permute(0, 2, 1)  # b,hw,c
        k = self.k(x_norm).reshape(b, c, -1)  # b,c,hw
        v = self.v(x_norm).reshape(b, c, -1).permute(0, 2, 1)
        attn = torch.bmm(q, k) * (c ** (-0.5))
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj(out)
        return x + out


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head self-attention pour qualité professionnelle"""
    def __init__(self, ch, num_heads=8):
        super().__init__()
        assert ch % num_heads == 0, f"channels {ch} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)  # Q, K, V en un seul conv
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.qkv(x_norm)  # b, 3*c, h, w
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, b, heads, hw, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # b, heads, hw, hw
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # b, heads, hw, head_dim
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        
        out = self.proj(out)
        return x + out


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.0,
        image_size=64,
        num_heads=1,  # Nouveau paramètre
        use_scale_shift_norm=False,  # Nouveau paramètre
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.image_size = image_size
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm

        # Time embedding
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_dim = time_dim

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down sampling blocks
        self.down = nn.ModuleList()
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(
                    ResidualBlock(
                        ch, out_ch, 
                        temb_dim=time_dim, 
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                )
                ch = out_ch
            
            # Add attention at specified resolutions
            if (image_size // (2 ** i)) in attention_resolutions:
                if num_heads > 1:
                    self.down.append(MultiHeadAttentionBlock(ch, num_heads=num_heads))
                else:
                    self.down.append(AttentionBlock(ch))
            
            # Downsample (except last level)
            if i != len(channel_mult) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        # Middle blocks
        self.mid1 = ResidualBlock(
            ch, ch, 
            temb_dim=time_dim, 
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm
        )
        if num_heads > 1:
            self.mid_attn = MultiHeadAttentionBlock(ch, num_heads=num_heads)
        else:
            self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResidualBlock(
            ch, ch, 
            temb_dim=time_dim, 
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm
        )

        # Up sampling blocks
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            # Upsample FIRST (except for first iteration at lowest resolution)
            if i != len(channel_mult) - 1:
                self.up.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            
            for _ in range(num_res_blocks):
                self.up.append(
                    ResidualBlock(
                        ch + out_ch, out_ch, 
                        temb_dim=time_dim, 
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                )
                ch = out_ch
            
            # Add attention at specified resolutions
            if (image_size // (2 ** i)) in attention_resolutions:
                if num_heads > 1:
                    self.up.append(MultiHeadAttentionBlock(ch, num_heads=num_heads))
                else:
                    self.up.append(AttentionBlock(ch))

        # Output layers
        self.norm_out = nn.GroupNorm(8, ch)
        self.act = SiLU()
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, timesteps):
        """
        x: [b, c, h, w] values in [-1,1]
        timesteps: tensor shape (b,) dtype long
        """
        # Time embedding
        temb = timestep_embedding(timesteps, self.model_channels).to(x.device)
        temb = self.time_mlp(temb)

        # Track skip connections
        hs = []
        h = self.conv_in(x)
        hs.append(h)

        # Downsampling
        for layer in self.down:
            if isinstance(layer, ResidualBlock):
                h = layer(h, temb)
                hs.append(h)
            elif isinstance(layer, (AttentionBlock, MultiHeadAttentionBlock)):
                h = layer(h)
                # Ne pas ajouter aux skip connections après attention
            else:  # Conv downsample
                h = layer(h)
                # Ne pas ajouter aux skip connections ici

        # Middle
        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        # Upsampling
        for layer in self.up:
            if isinstance(layer, nn.ConvTranspose2d):  # Upsample FIRST
                h = layer(h)
            elif isinstance(layer, ResidualBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, temb)
            elif isinstance(layer, (AttentionBlock, MultiHeadAttentionBlock)):
                h = layer(h)

        # Output
        h = self.norm_out(h)
        h = self.act(h)
        return self.conv_out(h)


# Test function pour vérifier les paramètres
def count_parameters(model):
    """Compte le nombre de paramètres entraînables"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("="*80)
    print("TEST DES CONFIGURATIONS")
    print("="*80)
    
    # Test Config1 (Original)
    model1 = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        num_res_blocks=2,
        attention_resolutions=[16],
        dropout=0.1,
        image_size=128,
        num_heads=1,
        use_scale_shift_norm=False
    )
    print(f"\n[CONFIG1] Paramètres: {count_parameters(model1):,} ({count_parameters(model1)/1e6:.1f}M)")
    
    # Test Config3 (Optimal)
    model3 = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=[1, 2, 3, 4],
        num_res_blocks=2,
        attention_resolutions=[16, 32],
        dropout=0.1,
        image_size=128,
        num_heads=1,
        use_scale_shift_norm=False
    )
    print(f"[CONFIG3] Paramètres: {count_parameters(model3):,} ({count_parameters(model3)/1e6:.1f}M)")
    
    # Test ConfigPro (Votre cible)
    model_pro = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=160,
        channel_mult=[1, 2, 3, 4],
        num_res_blocks=3,
        attention_resolutions=[8, 16, 32],
        dropout=0.1,
        image_size=128,
        num_heads=8,
        use_scale_shift_norm=True
    )
    print(f"[CONFIG PRO] Paramètres: {count_parameters(model_pro):,} ({count_parameters(model_pro)/1e6:.1f}M)")
    
    # Test forward pass
    print("\n" + "="*80)
    print("TEST FORWARD PASS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pro = model_pro.to(device)
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps: {t.shape}")
    
    with torch.no_grad():
        out = model_pro(x, t)
    
    print(f"Output shape: {out.shape}")
    
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"VRAM utilisée: {vram:.2f} GB")
    
    print("\n✅ Tests réussis!")
    print("="*80)