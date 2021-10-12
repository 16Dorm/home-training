import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        #assert transform is not None, "Set the transform on train set"
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        label = torch.tensor(self.df['label'].iloc[index])
        image = Image.open(self.df['image_path'].iloc[index])
        image = np.array(image.convert("RGB"))
        if self.transform:
            
            transformed_image = self.transform(image=image)#['image']
            image = transformed_image['image']
        return image, label
