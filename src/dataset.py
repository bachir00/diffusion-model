import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


def denormalize(x):
    # x in [-1,1] -> [0,1]
    return (x + 1) * 0.5


class CatDataset(Dataset):
    def __init__(self, data_dir, image_size=64, augment=False):
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('png','jpg','jpeg'))]
        self.files = sorted(files)
        self.image_size = image_size
        self.augment = augment
        self.transform = self._build_transform()

    def _build_transform(self):
        transforms = [T.Resize((self.image_size, self.image_size)), T.CenterCrop(self.image_size)]
        if self.augment:
            transforms = [T.RandomHorizontalFlip()] + transforms
        transforms += [T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        return T.Compose(transforms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img)
