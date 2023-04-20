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

class Sampler:

    def __init__(self, dataset):
        # get the class labels from the dataset
        y = np.concatenate([[data[1].numpy()] for data in dataset])
        # compute the class weights
        class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y])
        samples_weight = torch.from_numpy(samples_weight)
        weight = torch.from_numpy(weight)
        # use the class weights to create a WeightedRandomSampler
        self.weight = weight.float()
        self.sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight, num_samples=len(dataset), replacement=True)

    def get_weight(self):
        return self.weight
    
    def get_sampler(self):
        return self.sampler

# Focal loss from:
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8

class FocalLoss(torch.nn.Module):
    
    def __init__(self, weight=None, gamma=2., reduction='none'):
        torch.nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = torch.nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return torch.nn.functional.nll_loss(((1 - prob) ** self.gamma) * log_prob,
                                            target_tensor,
                                            weight = self.weight,
                                            reduction = self.reduction)


