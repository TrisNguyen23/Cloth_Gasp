import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2

class GraspDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # scale keypoints tương ứng
        w, h = image.shape[1], image.shape[0]
        keypoints = row[1:].values.astype('float32')
        keypoints[0::2] *= (224 / w)
        keypoints[1::2] *= (224 / h)

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        target = torch.tensor(keypoints, dtype=torch.float32)

        return image, target
