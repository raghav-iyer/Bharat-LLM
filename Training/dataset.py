import torch
from torch.utils.data import Dataset

class MultilingualPretrainingDataset(Dataset):
    def __init__(self, tokenized_files, seq_len=512):
        self.tokenized_data = []
        for file in tokenized_files:
            self.tokenized_data.extend(torch.load(file))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        token_ids = self.tokenized_data[idx]
        if len(token_ids) > self.seq_len:
            token_ids = token_ids[:self.seq_len]
        else:
            token_ids += [0] * (self.seq_len - len(token_ids))  # Pad with [PAD] token
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, target_ids
