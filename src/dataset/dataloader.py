from pathlib import Path
import os

import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, data: str | Path, batch_size, sequence_length, encoder_type='gpt2'):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        if os.path.isfile(data):
            with Path(data).open('r') as f_r:
                text = f_r.read()
        else:
            text = data

        enc = tiktoken.get_encoding(encoder_type)
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = 0

    def _next_batch(self):
        batch_size, sequence_length = self.batch_size, self.sequence_length
        buf = self.tokens[self.current_position: self.current_position + batch_size*sequence_length+1]
        x = buf[:-1].view(batch_size, sequence_length)
        y = buf[1:].view(batch_size, sequence_length)
        self.current_position += batch_size*sequence_length

        if self.current_position + batch_size * sequence_length > len(self.tokens):
            self.current_position = 0
        return x, y
