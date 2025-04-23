from pathlib import Path

import tiktoken
import torch
from huggingface_hub import snapshot_download
from datasets import load_dataset

from pytorch_lightning import LightningDataModule
from torch.utils.data import IterableDataset, DataLoader, BatchSampler


class IterableFixedSequenceLengthDataset(IterableDataset):
    def __init__(self, dataset, length):
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.remaining = []

    def __iter__(self):
        buf = self.remaining
        for data in self.dataset:
            buf += data["input_ids"]
            if len(buf) >= self.length+1:
                x = buf[:self.length]
                y = buf[1:self.length + 1]
                self.remaining = buf[self.length:]
                buf = self.remaining
                yield torch.tensor(x), torch.tensor(y)


class FineWebDatamodule(LightningDataModule):
    def __init__(self, local_dir="", sample="10BT", enc_type="gpt2",
                 batch_size=2):
        super().__init__()
        self.save_hyperparameters({"batch_size": batch_size, "dataset": sample})
        self.local_dir = Path(local_dir)
        self.sample = sample
        self.pattern = f"sample/{sample}/*"
        self.enc = tiktoken.get_encoding(enc_type)

        self.fineweb_train = None

    def prepare_data(self):
        folder = snapshot_download(
            "HuggingFaceFW/fineweb",
            repo_type="dataset",
            local_dir=self.local_dir,
            allow_patterns=self.pattern)

    def _get_tokenized_dataset(self, split="train"):
        eot = self.enc._special_tokens['<|endoftext|>']
        data_path = self.local_dir.joinpath("sample").joinpath(self.sample)
        dataset_text = load_dataset(path=str(data_path), split=split, streaming=True)
        dataset = dataset_text.map(lambda x: {"input_ids": [[eot] + self.enc.encode_ordinary(i) for i in x["text"]]}, batched=True, batch_size=2)
        dataset = IterableFixedSequenceLengthDataset(dataset, 1024)
        return dataset

    def setup(self, stage=None):
        if stage == "fit":
            self.fineweb_train = self._get_tokenized_dataset("train")


    def train_dataloader(self):
        return DataLoader(self.fineweb_train, batch_size=self.hparams.batch_size)


if __name__ == '__main__':
    dm = FineWebDatamodule(local_dir="/home/gerhard/PycharmProjects/gpt-2/data/fineweb", sample="10BT")
    dm.prepare_data()
    dm.setup("fit")

    train_loader = dm.train_dataloader()
