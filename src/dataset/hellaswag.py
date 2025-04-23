from pathlib import Path
import json
from typing import Tuple, List

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import tqdm
import requests
import tiktoken


class HellaSwagDataset(Dataset):
    def __init__(self, data: List[Tuple[Tensor, Tensor, int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return data in form of 4 X N ctx, 4 X N mask and int label
        return self.data[idx]


class HellaSwagDataModule(LightningDataModule):
    def __init__(self, data_dir: str | Path = "", enc_type: str = "gpt2",
                 batch_size=2, num_workers=4):
        super().__init__()
        self.save_hyperparameters({"batch_size": batch_size})
        self.data_dir = Path(data_dir)
        self.enc = tiktoken.get_encoding(enc_type)
        self.num_workers=num_workers

        self.data_paths = {
            split: self.data_dir.joinpath(f"hellaswag_{split}.jsonl") for split in ("train", "val", "test")
        }

        self.remote_source = {
            "train": "https://raw.githubusercontent.com/rowanz/hellaswag/refs/heads/master/data/hellaswag_train.jsonl",
            "val": "https://raw.githubusercontent.com/rowanz/hellaswag/refs/heads/master/data/hellaswag_val.jsonl",
            "test": "https://raw.githubusercontent.com/rowanz/hellaswag/refs/heads/master/data/hellaswag_test.jsonl"
        }

    def _download_data(self, partition):
        resp = requests.get(self.remote_source[partition], stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(self.data_paths[partition], "wb") as f_b, tqdm.tqdm(
            total=total,
            desc=partition,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024):
                size = f_b.write(chunk)
                bar.update(size)

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        for partition, path in self.data_paths.items():
            if not path.exists():
                self._download_data(partition)


    def _render_example(self, example):
        ctx = example["ctx"]
        endings = example["endings"]
        label = example["label"]

        ctx_tokens = self.enc.encode(ctx)
        masks_enc = []
        tokens_enc = []
        for ending in endings:
            end_tokens = self.enc.encode(' ' + ending)
            tokens_enc.append(ctx_tokens+end_tokens)
            mask = [0] * len(ctx_tokens) + [1] * len(end_tokens)
            masks_enc.append(mask)

        max_len = max([len(i) for i in tokens_enc])
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        masks = torch.zeros((4, max_len), dtype=torch.long)

        for i, ending_enc in enumerate(tokens_enc):
            tokens[i, :len(ending_enc)] = torch.tensor(ending_enc)
            masks[i, :len(ending_enc)] = torch.tensor(masks_enc[i])

        return tokens, masks, label


    def _load_data(self, partition):
        with open(self.data_paths[partition], "r") as f:
            data = [json.loads(i) for i in f.readlines()]
        data = [self._render_example(i) for i in data]
        return data


    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_data = self._load_data("train")
            self.hellaswag_train = HellaSwagDataset(train_data)
            val_data = self._load_data("val")
            self.hellaswag_val = HellaSwagDataset(val_data)
        if stage == "validate":
            val_data = self._load_data("val")
            self.hellaswag_val = HellaSwagDataset(val_data)
        if stage == "test":
            self.test_data = self._load_data("test")
            self.hellaswag_test = HellaSwagDataset(self.test_data)
        if stage == "predict":
            pass

    @staticmethod
    def pad(data: list[Tuple[Tensor, Tensor, int]]):
        tokens = [i[0].transpose(0, 1) for i in data]
        masks = [i[1].transpose(0, 1) for i in data]
        labels = [i[2] for i in data]

        tokens = pad_sequence(tokens, batch_first=True, padding_value=0).transpose(-2, -1)
        masks = pad_sequence(masks, batch_first=True, padding_value=0).transpose(-2, -1)
        tokens = tokens.contiguous().view(-1, tokens.shape[-1])
        masks = masks.contiguous().view(-1, masks.shape[-1])
        labels = torch.as_tensor(labels)
        return tokens, masks, labels

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.hellaswag_train, batch_size=self.hparams.batch_size, collate_fn=self.pad, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.hellaswag_val, batch_size=self.hparams.batch_size, collate_fn=self.pad, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.hellaswag_test, batch_size=self.hparams.batch_size, collate_fn=self.pad, num_workers=self.num_workers)



if __name__ == "__main__":
    dm = HellaSwagDataModule("../../data/hellaswag/")
    dm.prepare_data()
    dm.setup("fit")
    dataloader = dm.train_dataloader()
    for batch in dataloader:
        break
