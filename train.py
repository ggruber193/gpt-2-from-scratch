import math
import time
import torch
import tqdm
from torch.nn import functional as F
from peft import TaskType
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ThroughputMonitor
from torch.utils.data import DataLoader
import peft

from src.dataset.fineweb_edu import FineWebDatamodule
from src.model.gpt import GPT2
from src.utils.scheduler import GPTLrScheduler
from src.dataset.dataset import SimpleTextDataset
from src.model.lightning import GPTLightning
from src.dataset.hellaswag import HellaSwagDataModule


if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    # load_dotenv()
    # databricks_api_key = os.getenv("DATABRICKS_API_KEY")
    # databricks_uri = os.getenv("DATABRICKS_URI")
    # os.environ["DATABRICKS_HOST"] = databricks_uri
    # os.environ["DATABRICKS_TOKEN"] = databricks_api_key
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.set_float32_matmul_precision('high')

    B = 2
    T = 1024
    grad_accum_steps = 64

    dm_hellaswag = HellaSwagDataModule("../data/hellaswag", num_workers=4)
    dm_hellaswag.prepare_data()
    dm_hellaswag.setup("fit")

    dm_fineweb = FineWebDatamodule("../data/fineweb", sample="10BT")
    dm_fineweb.prepare_data()
    dm_fineweb.setup("fit")

    model = GPT2.from_pretrained("gpt2")

    lora_config = peft.LoraConfig(
        r=16,
        target_modules=[i for i, _ in model.named_modules() if 'c_attn' in i],
        lora_alpha=32,
        lora_dropout=0.5
    )
    lora_model = peft.get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    model = GPTLightning.from_model(lora_model)

    #logger = MLFlowLogger(run_name="finetune-gpt2-hellaswag", tracking_uri=databricks_uri)
    logger = TensorBoardLogger(save_dir="lightning_logs", name="finetune-gpt2-hellaswag")

    trainer = Trainer(accelerator="auto", max_epochs=10, accumulate_grad_batches=grad_accum_steps,
                      precision="bf16-mixed", logger=logger)
    trainer.fit(model, train_dataloaders=dm_fineweb.train_dataloader(), val_dataloaders=dm_hellaswag.val_dataloader())
