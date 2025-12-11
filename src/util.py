import torch
import os


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # replace model params by EMA params
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

def save_image_grid(images, path, nrow=4):
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(0,1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil = to_pil_image(grid)
    pil.save(path)
