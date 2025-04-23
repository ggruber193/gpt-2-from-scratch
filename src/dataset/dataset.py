from torch.utils.data import Dataset
import os
from pathlib import Path

import tiktoken
import torch

class SimpleTextDataset(Dataset):
    def __init__(self, data, sequence_length, encoder_type="gpt2"):
        self.sequence_length = sequence_length

        if os.path.isfile(data):
            with Path(data).open('r') as f_r:
                text = f_r.read()
        else:
            text = data

        enc = tiktoken.get_encoding(encoder_type)
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

    def __len__(self):
        return self.tokens.shape[0] // self.sequence_length

    def __getitem__(self, idx):
        buf = self.tokens[idx*self.sequence_length: (idx+1)*self.sequence_length+1]
        x = buf[:-1]
        y = buf[1:]
        return x, y


if __name__ == '__main__':
    dataset = SimpleTextDataset("/home/gerhard/PycharmProjects/gpt-2/data/input.txt", 1024)
    print(len(dataset))
    print(dataset[0])
