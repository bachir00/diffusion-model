# 🎨 Modèle de Diffusion - Documentation du Code Source

Ce dossier contient l'implémentation complète d'un **modèle de diffusion (DDPM - Denoising Diffusion Probabilistic Model)** pour la génération d'images, optimisé pour l'entraînement sur GPU.

---

## 📁 Structure des Fichiers

### Fichiers Principaux

| Fichier | Description |
|---------|-------------|
| **`config.py`** | Configuration centralisée du modèle et de l'entraînement |
| **`model.py`** | Architecture U-Net avec attention et blocs résiduels |
| **`diffusion.py`** | Implémentation du processus de diffusion DDPM |
| **`train.py`** | Script d'entraînement principal avec optimisations |
| **`dataset.py`** | Chargement et prétraitement des données |
| **`generate.py`** | Génération d'images depuis un checkpoint entraîné |
| **`sample.py`** | Échantillonnage rapide d'images depuis un checkpoint |
| **`schedules.py`** | Différents schedules de bruit (linear, cosine, quadratic) |
| **`util.py`** | Utilitaires (EMA, sauvegarde d'images) |

### Dossiers

- **`checkpoints128_pro/`** : Checkpoints du modèle entraîné (128×128)
- **`samples128_pro/`** : Échantillons générés pendant l'entraînement

---

## 🧠 Architecture du Modèle

### 1. **U-Net (`model.py`)**

Le cœur du système est un **U-Net** (architecture encoder-decoder avec skip connections) :

#### Composants Principaux

```
Input (3×H×W) 
    ↓
[Timestep Embedding] → Encodage sinusoïdal du temps
    ↓
[Encoder] 
    • ResidualBlocks (avec time conditioning)
    • AttentionBlocks (self-attention)
    • Downsampling (réduction résolution)
    ↓
[Middle] 
    • ResidualBlock + Attention + ResidualBlock
    ↓
[Decoder]
    • Upsampling (augmentation résolution)
    • ResidualBlocks + Skip Connections
    • AttentionBlocks
    ↓
Output (3×H×W) → Prédiction du bruit
```

#### Classes Importantes

- **`SiLU`** : Activation Swish (x * sigmoid(x))
- **`timestep_embedding()`** : Encodage positionnel sinusoïdal du timestep
- **`ResidualBlock`** : 
  - Bloc résiduel avec normalisation de groupe
  - Injection du timestep via projection linéaire
  - Support du scale-shift normalization (optionnel)
- **`AttentionBlock`** : Self-attention spatial (single-head ou multi-head)
- **`UNet`** : Architecture complète avec skip connections

**Paramètres clés** :
- `model_channels` : Nombre de canaux de base (ex: 192)
- `channel_mult` : Multiplicateurs de canaux par niveau (ex: [1,2,3,4])
- `num_res_blocks` : Nombre de blocs résiduels par niveau
- `attention_resolutions` : Résolutions où appliquer l'attention (ex: [16])

---

### 2. **Processus de Diffusion (`diffusion.py`)**

Implémentation du **DDPM (Denoising Diffusion Probabilistic Model)** :

#### Forward Process (ajout de bruit)
```python
x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
```
où :
- `x_0` : image originale
- `x_t` : image bruitée au timestep t
- `ε` : bruit gaussien
- `α̅_t` : produit cumulé des alphas

#### Reverse Process (débruitage)
Le modèle apprend à prédire le bruit `ε` pour reconstituer l'image progressivement.

**Méthodes principales** :
- `q_sample()` : Ajoute du bruit à une image (forward)
- `p_mean_variance()` : Calcule la distribution pour le débruitage (reverse)
- `forward()` : Calcule la loss d'entraînement (MSE sur le bruit prédit)
- `sample()` : Génère de nouvelles images depuis du bruit pur

---

### 3. **Schedules de Bruit (`schedules.py`)**

Trois types de schedules pour contrôler l'ajout progressif de bruit :

| Schedule | Description | Usage |
|----------|-------------|-------|
| **Linear** | Augmentation linéaire de β_start à β_end | Simple, stable |
| **Cosine** | Suit une courbe cosinus (plus lent au début) | **Recommandé** - meilleure qualité |
| **Quadratic** | Augmentation quadratique | Compromis |

```python
betas = get_beta_schedule("cosine", timesteps=1000)
```

---

## 🔧 Configuration (`config.py`)

Le fichier `config.py` contient plusieurs classes de configuration adaptées à différentes contraintes :

### Exemple : `DiffusionConfig` (128×128 optimisé)

```python
class DiffusionConfig:
    # DONNÉES
    data_dir = "../data/train/cats_cleaned/good"
    image_size = 128
    in_channels = 3
    out_channels = 3
    
    # MODÈLE (U-Net)
    model_channels = 192           # Canaux de base
    channel_mult = [1, 2, 3, 4]    # 4 niveaux de résolution
    num_res_blocks = 2             # Blocs par niveau
    attention_resolutions = [16]   # Attention à 16×16
    dropout = 0.1
    
    # DIFFUSION
    timesteps = 1000               # Nombre d'étapes de diffusion
    beta_schedule = "cosine"       # Type de schedule
    
    # ENTRAÎNEMENT
    batch_size = 8
    num_epochs = 500
    learning_rate = 2e-4
    num_workers = 4
    
    # OPTIMISATIONS
    use_fp16 = False               # Mixed precision (économie VRAM)
    gradient_accumulation_steps = 1
    gradient_clip = 1.0
    ema_decay = 0.999             # Exponential Moving Average
```

**Notes importantes** :
- Les configurations commentées montrent l'évolution des paramètres testés
- Ajustez `model_channels` et `attention_resolutions` selon votre VRAM
- Plus `timesteps` est élevé, meilleure est la qualité (mais plus lent)

---

## 🚀 Entraînement (`train.py`)

### Fonctionnalités

#### 1. **Optimisations GPU**
- **Mixed Precision (FP16)** : Réduit la VRAM de ~40%
- **Gradient Accumulation** : Simule de plus gros batch_size
- **cuDNN Benchmark** : Optimisation automatique des kernels
- **TF32** : Activé automatiquement sur GPU Ampere (RTX 30/40)
- **Pin Memory** : Transfert CPU→GPU plus rapide

#### 2. **EMA (Exponential Moving Average)**
Maintient une version lissée des poids du modèle pour de meilleures générations :
```python
ema = EMA(unet, decay=0.999)
ema.update()  # Pendant l'entraînement
ema.apply_shadow()  # Pour la génération
```

#### 3. **Checkpoint & Sampling**
- Sauvegarde automatique tous les N epochs
- Génération d'échantillons pour suivre la progression
- Sauvegarde du meilleur modèle selon la loss

#### 4. **Learning Rate Warmup**
Augmentation progressive du learning rate au début pour stabiliser l'entraînement.

### Boucle d'Entraînement

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass avec mixed precision
        with autocast(device_type='cuda', enabled=use_fp16):
            loss = ddpm(batch)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping + optimization step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        ema.update()
```

### Reprise d'Entraînement

Le script supporte la reprise depuis un checkpoint :
```python
train(resume_from_checkpoint="checkpoints/checkpoint_epoch_0100.pt")
```

---

## 📊 Dataset (`dataset.py`)

### Classe `CatDataset`

Charge et prétraite les images :

```python
dataset = CatDataset(
    data_dir="../data/train/cats",
    image_size=128,
    augment=True  # Active l'augmentation de données
)
```

**Transformations appliquées** :
1. Resize vers la taille cible
2. Center Crop
3. Random Horizontal Flip (si `augment=True`)
4. Conversion en Tensor
5. Normalisation dans [-1, 1] : `(x - 0.5) / 0.5`

**Fonction `denormalize()`** : Reconvertit de [-1,1] vers [0,1] pour l'affichage.

---

## 🎨 Génération (`generate.py` & `sample.py`)

### Génération d'Images

```python
# Avec generate.py (complet)
generate_images(
    checkpoint_path="checkpoints/best_model.pt",
    num_images=16,
    output_dir="generated",
    device="cuda"
)

# Avec sample.py (rapide)
sample_from_checkpoint(
    ckpt_path="checkpoints/final_unet.pt",
    out_path="samples/output.png",
    num_samples=16
)
```

**Différences** :
- `generate.py` : Plus complet, gère plusieurs formats de sortie
- `sample.py` : Plus simple, pour tests rapides

### Processus de Génération

1. Commence avec du bruit pur : `x_T ~ N(0, I)`
2. Pour chaque timestep t de T à 0 :
   - Prédit le bruit avec le U-Net
   - Calcule `x_{t-1}` en retirant le bruit prédit
3. Retourne `x_0` (image finale)

---

## 🛠️ Utilitaires (`util.py`)

### Classe `EMA`

Maintient une moyenne mobile exponentielle des poids :
```python
ema = EMA(model, decay=0.9999)
ema.update()          # Met à jour la moyenne
ema.apply_shadow()    # Applique les poids EMA
ema.restore()         # Restaure les poids originaux
```

**Avantage** : Génère des images plus stables et de meilleure qualité.

### Fonction `save_image_grid()`

Sauvegarde une grille d'images :
```python
save_image_grid(
    images,                 # Tensor [N, C, H, W]
    path="output.png",
    nrow=4                  # Images par ligne
)
```

---

## 💡 Utilisation Pratique

### 1. Entraîner un Modèle

```bash
# Depuis le dossier src/
python train.py
```

### 2. Reprendre un Entraînement

Modifier `train.py` :
```python
if __name__ == '__main__':
    train(resume_from_checkpoint="checkpoints/checkpoint_epoch_0100.pt")
```

### 3. Générer des Images

```bash
python generate.py
```

### 4. Ajuster la Configuration

Éditer `config.py` et changer la classe active :
```python
# Utiliser DiffusionConfig au lieu de DiffusionConfig1
from config import DiffusionConfig
```

---

## 📈 Métriques & Suivi

### Logs d'Entraînement

Le fichier `training.log` contient :
- Loss par epoch/step
- Vitesse d'entraînement (images/sec)
- Utilisation mémoire
- Temps par epoch

### Visualisation

Les échantillons dans `samples128_pro/` permettent de :
- Suivre la progression visuelle
- Détecter l'overfitting
- Comparer différentes configurations

---

## ⚙️ Optimisations & Astuces

### Pour Réduire la VRAM

1. **Diminuer `batch_size`** : 8 → 4 ou 2
2. **Activer FP16** : `use_fp16 = True`
3. **Réduire `model_channels`** : 192 → 128 ou 96
4. **Moins d'attention** : `attention_resolutions = [16]` au lieu de `[16, 8]`
5. **Moins de niveaux** : `channel_mult = [1,2,3]` au lieu de `[1,2,3,4]`

### Pour Améliorer la Qualité

1. **Plus de timesteps** : 1000 → 1500 ou 2000
2. **Schedule cosine** : Meilleur que linear
3. **EMA decay élevé** : 0.9999 au lieu de 0.999
4. **Plus de données** : Dataset plus large et varié
5. **Plus d'epochs** : Entraîner plus longtemps

### Pour Accélérer l'Entraînement

1. **cuDNN benchmark** : `cudnn_benchmark = True`
2. **Persistent workers** : Dans DataLoader
3. **Pin memory** : `pin_memory = True`
4. **Gradient accumulation** : Si batch_size limité

---

## 🔍 Points Techniques Avancés

### Skip Connections dans U-Net

Les skip connections relient l'encoder au decoder :
```python
# Encoder: sauvegarde les features
hs.append(h)

# Decoder: récupère et concatène
skip = hs.pop()
h = torch.cat([h, skip], dim=1)
```

**Pourquoi ?** Préserve les détails haute fréquence perdus lors du downsampling.

### Time Conditioning

Le timestep est injecté dans chaque ResidualBlock :
```python
temb = timestep_embedding(t, dim)     # Encodage sinusoïdal
temb = time_mlp(temb)                 # Projection MLP
h = h + temb_proj(temb)[:,:,None,None]  # Ajout spatial
```

**Pourquoi ?** Le modèle doit savoir à quel niveau de bruit il travaille.

### Attention Mechanism

Self-attention pour capturer les relations spatiales :
```python
Q = q(x), K = k(x), V = v(x)
Attention = softmax(Q·K^T / √d) · V
```

**Coût** : Quadratique en résolution → appliqué seulement à basses résolutions (ex: 16×16).

---

## 🐛 Troubleshooting

| Problème | Solution |
|----------|----------|
| **Out of Memory (CUDA OOM)** | Réduire batch_size, activer FP16, diminuer model_channels |
| **Loss ne descend pas** | Vérifier learning rate, augmenter warmup_steps, vérifier données |
| **Images floues** | Augmenter timesteps, utiliser schedule cosine, entraîner plus longtemps |
| **Mode collapse** | Augmenter dropout, vérifier diversité du dataset, réduire learning rate |
| **Artefacts en damier** | Remplacer ConvTranspose2d par Upsample + Conv2d |

---

## 📚 Références Théoriques

- **Paper DDPM** : [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- **Improved DDPM** : [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
- **Architecture U-Net** : [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## 🎯 Résumé du Pipeline

```
1. Configuration (config.py)
   ↓
2. Chargement données (dataset.py)
   ↓
3. Création modèles (model.py + diffusion.py)
   ↓
4. Boucle d'entraînement (train.py)
   • Forward: ajout de bruit + prédiction
   • Backward: MSE loss + optimisation
   • EMA update + sampling périodique
   ↓
5. Sauvegarde checkpoints
   ↓
6. Génération finale (generate.py)
```

---

## 📝 Notes Finales

Ce code est **production-ready** avec :
- ✅ Support GPU optimisé (FP16, gradient accumulation)
- ✅ Reprise d'entraînement robuste
- ✅ Logging détaillé
- ✅ EMA pour stabilité
- ✅ Configurations multiples

**Configuration recommandée pour RTX 4070 8GB** :
- `image_size = 128`
- `model_channels = 192`
- `batch_size = 8`
- `attention_resolutions = [16]`
- `use_fp16 = True` (si besoin)

---

