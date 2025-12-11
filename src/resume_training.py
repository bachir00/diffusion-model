"""
Script pour reprendre l'entraînement depuis le dernier checkpoint
"""

import os
import glob
from train import train

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Trouve le dernier checkpoint disponible"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    
    if not checkpoints:
        print("❌ Aucun checkpoint trouvé!")
        return None
    
    # Trier par numéro d'epoch
    checkpoints.sort()
    latest = checkpoints[-1]
    
    print(f"📌 Dernier checkpoint trouvé: {latest}")
    return latest

if __name__ == '__main__':
    # Trouver le dernier checkpoint
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"\n🔄 Reprise de l'entraînement depuis: {latest_checkpoint}\n")
        train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("\n🆕 Aucun checkpoint trouvé, démarrage d'un nouvel entraînement...\n")
        train()
