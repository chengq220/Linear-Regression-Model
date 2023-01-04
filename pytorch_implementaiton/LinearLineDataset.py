import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import TensorDataset

"""Custom dataset class"""

class LinearLineDataset(Dataset):
    def __init__(self, csv_file_dir, transform=None):
        """
        :param csv_file_dir: the path to the csv file
        :param transform: optional transform to be applied on dataset

        """
        self.dataset = pd.read_csv(csv_file_dir)
        self.features, self.target = self.minMaxNormalization(self.dataset)

    def minMaxNormalization(self,dataset):
        x, y = dataset.iloc[:,0],dataset.iloc[:,1]
        x_min,x_max = dataset.iloc[:,0].min(),dataset.iloc[:,0].max()
        x = (x - x_min)/(x_max-x_min)
        return x,y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = [[self.features[idx]],[self.target[idx]]]
        return torch.FloatTensor(sample)

    def __getdataset__(self):
        x = torch.from_numpy(self.features.to_numpy()).reshape(len(self.features),1).float()
        y = torch.from_numpy(self.target.to_numpy()).reshape(len(self.target), 1).float()
        return TensorDataset(x, y)
