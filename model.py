import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        probas = torch.nn.functional.softmax(logits)
        return probas
    

class CustomDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.x_data = np.load(x_file)
        self.y_data = pd.read_csv(y_file,usecols=['Domain'])
        self.encoded_labels = self.target_transform()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        x = torch.Tensor(self.x_data[idx])
        y = torch.tensor(self.encoded_labels[idx])

        return x, y
    
    def target_transform(self):
        mapping = { 'Application software':0,
                    'Documentation':1,
                    'Non-web libraries and frameworks':2,
                    'Software tools':3,
                    'System software':4,
                    'Web libraries and frameworks':5}

        return self.y_data['Domain'].map(mapping).to_numpy()
    

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
                
            # 1st hidden layer
            torch.nn.Linear(num_features, 768),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(768, 192),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(192, num_classes),
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits