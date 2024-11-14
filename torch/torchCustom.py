# Dataset
import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        X_data = pd.read_csv(self.data_dir + 'Reuters_X.txt',
                            sep=' ', header=None)
        y_data = pd.read_csv(self.data_dir + 'Reuters_Y.txt',
                            header=None)
        
        self.X = torch.FloatTensor(X_data.iloc[:, :].values)
        self.y = torch.LongTensor(y_data.iloc[:, 0].values).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = {'X':self.X[idx, :], 'y':self.y[idx, :]}

        if self.transform:
            data = self.transform(data)

        return data
    

# Loss
from Algorithms import TreeEmbedding
from Preprocessing import DataPreprocessing
from torch import nn
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self, tree):
        # calling the constructor of nn.Module
        super(CustomLoss, self).__init__() 

        self.tree = tree
        self.trees = TreeEmbedding(self.tree)
        self.Xi = torch.FloatTensor(self.trees.xi)


    def compute_xi_diff(self, y):
        paths = DataPreprocessing(self.tree).y_to_path(y)
        xi_diff = []

        for path in paths:
            path = path[path > 0]
            num_sib = [len(self.trees.nodes[node].siblings) for node in path]
            y_hat_ls = np.repeat(path, num_sib)

            y_tilde_ls = []
            for node in path:
                sib = [sib.id for sib in self.trees.nodes[node].siblings]
                y_tilde_ls += sib
            y_tilde_ls = np.array(y_tilde_ls)
           
            xi_diff.append(self.Xi[:, y_hat_ls-1] - self.Xi[:, y_tilde_ls-1])

        return xi_diff


    def forward(self, pred, true):
        pred = pred.unsqueeze(2)
        xi_diff = self.compute_xi_diff(true)
        n = len(xi_diff)

        loss = 0
        for i in range(n):
            Gm_i = torch.matmul(xi_diff[i].T, pred[i])
            hinge_i = torch.clamp(1-Gm_i, min=0)
            loss += torch.sum(hinge_i)
        loss /= n

        return loss
    


# Model
class CustomModel(nn.Module):
    def __init__(self, feature_length, output_size):
        super(CustomModel, self).__init__()
 
        self.hidden = nn.Linear(feature_length, 64, bias=False)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(64, output_size, bias=False)
 
    def forward(self, x):
        x = self.tanh(self.hidden(x))
        x = self.output(x)
 
        return x
    

    
# Model
class CustomPrediction(nn.Module):
    def __init__(self, tree):
        super(CustomPrediction, self).__init__()
 
        self.tree = tree
        self.trees = TreeEmbedding(self.tree)
        self.df = self.trees.df
        self.Xi = torch.FloatTensor(self.trees.xi)
 
    def forward(self, model, X):
        f = model(X)

        y_pred = torch.zeros((X.shape[0], self.tree.height+1), dtype=torch.long)
        l1 = self.df[self.df.parent == 0]['id'].to_list()
        l1 = [l-1 for l in l1]
        
        for s in range(X.shape[0]):
            fx = f[s]
            g = torch.matmul(self.Xi[:, l1].T, fx)
            y_pred[s, 1] = l1[torch.argmax(g).item()] + 1
            m = 1

            while not self.df[self.df.id == y_pred[s, m].item()]['isLeaf'].item():
                if m+1 >= y_pred.shape[1]:
                    break
                lm = self.df[self.df.parent == y_pred[s, m].item()]['id'].to_list()
                lm = [l-1 for l in lm]
                g = torch.matmul(self.Xi[:, lm].T, fx)
                y_pred[s, m+1] = lm[torch.argmax(g).item()] + 1
                m += 1

        return y_pred