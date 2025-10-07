from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np

class ImgDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = Path(path)
        self.images =  list((self.path).glob("img/*.png"))
        self.labels = list((self.path).glob("label/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("RGB"))

        if self.transform is not None:
            augmentation = self.transform(image=image, image1=label)
            image, label = augmentation['image'], augmentation['image1']
        
        return image, label 

        
