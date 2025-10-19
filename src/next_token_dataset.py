from torch.utils.data import Dataset
import torch

class TweetsDataset(Dataset):
    def __init__(self, data, targets, mask):
        self.data = data
        self.targets = targets 
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):       
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'mask': torch.tensor(self.mask[idx], dtype=torch.long)
        }
        
        
