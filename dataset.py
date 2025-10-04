from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL.Image import Image

class ImgDataset(Dataset):
    def __init__(self, path, is_eval=False):
        super().__init__()
        self.is_eval = is_eval
        self.path = Path(path)

        if not is_eval:
            self.images =  (self.path).glob("train/img/*.png")
            self.labels = (self.path).glob("train/label/*.png")
        else:
            self.images =  (self.path).glob("val/img/*.png")
            self.labels = (self.path).glob("val/label/*.png")

        self.transform = 


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        pil_img = Image.load(img_path)

        
