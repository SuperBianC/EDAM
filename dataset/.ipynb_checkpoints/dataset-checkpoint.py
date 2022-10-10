import torch
from torch.utils.data import Dataset

class SliceDataset(Dataset):
    def __init__(self, DictList, transform=None, target_transform=None):
        self.slices = DictList
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        SliceDict = {}
        tmp = self.slices[index]
        SliceDict['ids'] = torch.LongTensor(tmp['ids'])
        SliceDict['label'] = torch.scalar_tensor(tmp['label']).long()
        SliceDict['length'] = torch.scalar_tensor(tmp['length']).long()
        SliceDict['name'] = tmp['name']
        return SliceDict
    
