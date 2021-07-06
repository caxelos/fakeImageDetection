import torch, os
from PIL import Image
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform):
        self.main_dir = dataset_dir
        self.total_imgs = os.listdir(dataset_dir)
        self.transform=transform
    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    def __len__(self):
        return len(self.total_imgs)
    def __repr__(self):
        return self.__class__.__name__

