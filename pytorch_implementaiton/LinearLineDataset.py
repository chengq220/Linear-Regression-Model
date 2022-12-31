import torch
from torch.utils.data import Dataset
import pandas as pd


"""Custom dataset class"""

class LinearLineDataset(Dataset):
    def __init__(self, csv_file_dir, transform=None):
        """
        :param csv_file_dir: the path to the csv file
        :param transform: optional transform to be applied on dataset

        """
        self.dataset = pd.read_csv(csv_file_dir)
        self.features,self.target = self.minMaxNormalization(self.dataset)
        self.transform = transform

    def minMaxNormalization(self,dataset):
        x, y = dataset.iloc[:,0],dataset.iloc[:,1]
        x_min,x_max = dataset.iloc[:,0].min(),dataset.iloc[:,0].max()
        x = (x - x_min)/(x_max-x_min)
        return x,y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = [self.features[idx],self.target[idx]]
        return torch.FloatTensor(sample)

"""

testing
linedataset = LinearLineDataset("./linear_line.csv")
print(linedataset.__len__())
print(linedataset.__getitem__(250))

"""