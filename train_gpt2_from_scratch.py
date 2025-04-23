import math
import time
import torch
import tqdm
from torch.nn import functional as F
from peft import TaskType
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ThroughputMonitor
from torch.utils.data import DataLoader
import peft

from src.dataset.fineweb_edu import FineWebDatamodule
from inference import generate
from src.model.gpt import GPT2
from src.utils.scheduler import GPTLrScheduler
from src.dataset.dataset import SimpleTextDataset
from src.model.lightning import GPTLightning
from src.dataset.hellaswag import HellaSwagDataModule


if __name__ == '__main__':
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.set_float32_matmul_precision('high')

    B = 1
    T = 1024
    grad_accum_steps = 1

    dm_hellaswag = HellaSwagDataModule("../data/hellaswag", batch_size=B)
    dm_hellaswag.prepare_data()
    dm_hellaswag.setup("validate")

    dm_fineweb = FineWebDatamodule("../data/fineweb", sample="10BT", batch_size=B)
    dm_fineweb.prepare_data()
    dm_fineweb.setup("fit")

    model = GPTLightning.from_pretrained("gpt2")
    model.model.to("cuda")

    print(generate(model.model, "Hello I'm a language model,"))

    trainer = Trainer(accelerator="auto", max_epochs=1, accumulate_grad_batches=grad_accum_steps,
                      precision="bf16-mixed",limit_train_batches=10000)
    trainer.fit(model, train_dataloaders=dm_fineweb.train_dataloader())

    model.model.to("cuda")
    print(generate(model.model, "Hello I'm a language model,"))
