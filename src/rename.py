import os

def rename_files_to_cat(folder_path):
    """
    Renomme tous les fichiers d'un dossier en cat_1, cat_2, ..., cat_n
    en conservant leur extension.
    """
    # Lister les fichiers triés
    files = sorted(os.listdir(folder_path))

    index = 1
    for filename in files:
        old_path = os.path.join(folder_path, filename)

        # Ignorer les dossiers
        if not os.path.isfile(old_path):
            continue

        # Extraire extension
        extension = os.path.splitext(filename)[1]

        # Nouveau nom
        new_name = f"cat_{index}{extension}"
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"{filename} → {new_name}")

        index += 1

    print("✔️ Renommage terminé !")

if __name__ == "__main__":
    print("Renommage des fichiers...")
    folder_path = "../data/train/cats_cleaned/good"
    rename_files_to_cat(folder_path)
