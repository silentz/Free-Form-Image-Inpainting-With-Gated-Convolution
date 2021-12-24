import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
from wandb.sdk.data_types import Image
import pytorch_lightning as pl
from typing import Any, Dict, List

from src.collate import collate_fn, Batch
from src.models import Generator, Discriminator


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset: Dataset,
                       train_batch_size: int,
                       train_num_workers: int,
                       val_dataset: Dataset,
                       val_batch_size: int,
                       val_num_workers: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': collate_fn,
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
                'collate_fn': collate_fn,
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)


class Module(pl.LightningModule):

    def __init__(self, gen_optimizer_lr: float,
                       dis_optimizer_lr: float,
                       n_examples: int):
        super().__init__()
        self.gen_optimizer_lr = gen_optimizer_lr
        self.dis_optimizer_lr = dis_optimizer_lr
        self.n_examples = n_examples

        self.generator = Generator()
        self.discriminator = Discriminator()

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=self.gen_optimizer_lr)
        dis_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.dis_optimizer_lr)
        return [dis_optim, gen_optim]

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int) -> Dict[str, Any]:
        pass

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        pass

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pass
