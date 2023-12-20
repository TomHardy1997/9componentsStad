import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms  
from torch.utils.data.sampler import WeightedRandomSampler

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        df = df
        label_mapping = {
                'ADI': 0,
                'BAC': 1,
                'DEB': 2,
                'LYM': 3,
                'MUC': 4,
                'MUS': 5,
                'NOR': 6,
                'STR': 7,
                'TUM': 8
            }
        # import ipdb;ipdb.set_trace()
        df['label'] = df['class'].map(label_mapping)
        self.df = df
        self.transform = transform
        class_counts = self.df['label'].value_counts().sort_index()
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        self.sample_weights = weights[self.df['label'].values].to_numpy()
        self.sampler = WeightedRandomSampler(self.sample_weights, len(self.df))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        image = Image.open(path).convert('RGB')
        if image.size < (224, 224):
            raise ValueError(f"Image size is too small: {image.size}")
        label = self.df.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    df = pd.read_csv('shuffled_data.csv')
    dataset = CustomDataset(df,transform=transform)
    batch_size = 13500
    loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.sampler,num_workers=2)
    import ipdb;ipdb.set_trace()