from torch.utils.data import DataLoader, Dataset

class EmptyDataset(Dataset):
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError("This dataset is empty")