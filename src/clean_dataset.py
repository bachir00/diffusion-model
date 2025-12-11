"""
clean_dataset.py
Détecte images corrompues / low-res / floues / doublons et produit dossiers:
  dst/good, dst/bad, dst/duplicates, dst/review
Usage:
  python clean_dataset.py --src data/raw --dst data/cleaned
"""
import os
import argparse
from PIL import Image
import numpy as np
import cv2
import imagehash
from tqdm import tqdm
import shutil
import pandas as pd

def is_corrupted(path):
    try:
        im = Image.open(path)
        im.verify()
        return False
    except Exception:
        return True

def variance_of_laplacian_gray(path):
    # retourne la variance de Laplacian (score de netteté) pour une image
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def perceptual_hash(path):
    im = Image.open(path).convert("RGB")
    return imagehash.phash(im)

def resize_save(path, out_path, size=(128,128)):
    im = Image.open(path).convert("RGB")
    im = im.resize(size, resample=Image.LANCZOS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, quality=95)

def main(src, dst, min_width=64, min_height=64, min_file_size=8*1024,
         blur_thresh=50.0, hash_thresh=6):
    os.makedirs(dst, exist_ok=True)
    good_dir = os.path.join(dst, "good")
    bad_dir = os.path.join(dst, "bad")
    dup_dir = os.path.join(dst, "duplicates")
    review_dir = os.path.join(dst, "review")
    for d in (good_dir, bad_dir, dup_dir, review_dir):
        os.makedirs(d, exist_ok=True)

    records = []
    hashes = {}  # hash -> path

    files = [os.path.join(src,f) for f in os.listdir(src) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    for path in tqdm(files, desc="Scan images"):
        fname = os.path.basename(path)
        rec = {"path": path, "fname": fname, "corrupted": False, "width": None, "height": None,
               "filesize": os.path.getsize(path), "blur_score": None, "phash": None, "action": None}
        # check corrupted
        if is_corrupted(path):
            rec["corrupted"] = True
            rec["action"] = "corrupted"
            shutil.copy2(path, os.path.join(bad_dir, fname))
            records.append(rec)
            continue

        # open with PIL to check size
        try:
            im = Image.open(path)
            w,h = im.size
            rec["width"], rec["height"] = w, h
        except Exception:
            rec["corrupted"] = True
            rec["action"] = "corrupted"
            shutil.copy2(path, os.path.join(bad_dir, fname))
            records.append(rec)
            continue

        # low resolution or tiny file
        if w < min_width or h < min_height or rec["filesize"] < min_file_size:
            rec["action"] = "lowres_or_tiny"
            shutil.copy2(path, os.path.join(bad_dir, fname))
            records.append(rec)
            continue

        # blur score (Laplacian). Note: threshold depends on image size; adjust if needed
        try:
            blur = variance_of_laplacian_gray(path)
        except Exception:
            blur = 0.0
        rec["blur_score"] = blur
        if blur < blur_thresh:
            # move to review (or bad if definitely low)
            rec["action"] = "blurry"
            shutil.copy2(path, os.path.join(review_dir, fname))
            records.append(rec)
            continue

        # perceptual hash for duplicates
        try:
            ph = perceptual_hash(path)
        except Exception:
            ph = None
        rec["phash"] = str(ph)
        if ph is not None:
            found = None
            for hsh, pth in hashes.items():
                # distance Hamming between phashes
                dist = ph - hsh
                if dist <= hash_thresh:
                    found = pth
                    break
            if found:
                rec["action"] = "duplicate_of"
                rec["duplicate_of"] = found
                shutil.copy2(path, os.path.join(dup_dir, fname))
                records.append(rec)
                continue
            else:
                hashes[ph] = path

        # if everything OK -> resize and copy to good
        try:
            out_path = os.path.join(good_dir, fname)
            resize_save(path, out_path, size=(128,128))
            rec["action"] = "good"
        except Exception as e:
            rec["action"] = f"error_resizing:{e}"
            shutil.copy2(path, os.path.join(bad_dir, fname))
        records.append(rec)

    # # save CSV report
    # df = pd.DataFrame(records)
    # df.to_csv(os.path.join(dst, "cleaning_report.csv"), index=False)
    # print("Done. Report at:", os.path.join(dst, "cleaning_report.csv"))

if __name__ == "__main__":

    ######################### Command to run ###############################################
    # python clean_dataset.py --src D:/diffusion_model/data/train/cats --dst D:/diffusion_model/data/train/cats_cleaned

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--min_w", type=int, default=64)
    parser.add_argument("--min_h", type=int, default=64)
    parser.add_argument("--min_size", type=int, default=8*1024)
    parser.add_argument("--blur_thresh", type=float, default=50.0)
    parser.add_argument("--hash_thresh", type=int, default=6)
    args = parser.parse_args()
    main(args.src, args.dst, min_width=args.min_w, min_height=args.min_h,
         min_file_size=args.min_size, blur_thresh=args.blur_thresh, hash_thresh=args.hash_thresh)




