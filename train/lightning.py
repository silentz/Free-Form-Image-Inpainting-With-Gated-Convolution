import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
from wandb.sdk.data_types import Image
import pytorch_lightning as pl
from typing import Any, Dict, List

from src.collate import collate_fn, Batch
from src.models import Generator, Discriminator
from src.loss import recon_loss


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
                       dis_optimizer_lr: float):
        super().__init__()
        self.gen_optimizer_lr = gen_optimizer_lr
        self.dis_optimizer_lr = dis_optimizer_lr

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.automatic_optimization = False

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=self.gen_optimizer_lr)
        dis_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.dis_optimizer_lr)
        return [dis_optim, gen_optim]

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        dis_optim, gen_optim = self.optimizers()
        images = batch.images
        masks = batch.masks.unsqueeze(dim=1)

        X_coarse, X_recon, _ = self.generator(images, masks)
        X_complete = X_recon * masks + images * (1 - masks)

        # discriminator step
        X_disc_input = torch.cat([images, X_complete.detach()], dim=0)
        X_disc_masks = torch.cat([masks, masks], dim=0)
        X_disc_out = self.discriminator(X_disc_input, X_disc_masks)
        X_real, X_fake = torch.chunk(X_disc_out, chunks=2, dim=0)

        dis_real_loss = torch.mean(F.relu(1 - X_real))
        dis_fake_loss = torch.mean(F.relu(1 + X_fake))
        dis_loss = dis_real_loss + dis_fake_loss

        dis_optim.zero_grad()
        gen_optim.zero_grad()
        self.manual_backward(dis_loss, retain_graph=True)
        dis_optim.step()

        # generator step
        X_fake = self.discriminator(X_complete, masks)
        gen_gan_loss = -1 * torch.mean(X_fake)
        gen_rec_loss = recon_loss(images, X_coarse, X_recon, masks)
        gen_loss = gen_gan_loss + gen_rec_loss

        gen_optim.zero_grad()
        dis_optim.zero_grad()
        self.manual_backward(gen_loss)
        gen_optim.step()

        self.log('gen_gan_loss', gen_loss.item())
        self.log('gen_rec_loss', gen_rec_loss.item())
        self.log('gen_all_loss', gen_loss.item())
        self.log('disc_real_loss', dis_real_loss.item())
        self.log('disc_fake_loss', dis_fake_loss.item())
        self.log('disc_all_loss', dis_loss.item())

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        images = batch.images
        masks = batch.masks.unsqueeze(dim=1)

        X_coarse, X_recon, _ = self.generator(images, masks)
        X_complete = X_recon * masks + images * (1 - masks)

        return {
                'origin': images.detach().cpu().to(torch.float32),
                'mask': masks.detach().cpu().to(torch.float32),
                'coarse': X_coarse.detach().cpu().to(torch.float32),
                'complete': X_complete.detach().cpu().to(torch.float32),
            }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        n_batches = len(outputs)
        columns = ['origin', 'mask', 'coarse', 'complete']
        table = wandb.Table(columns=columns)

        for idx in range(n_batches):
            origins = outputs[idx]['origin']
            masks = outputs[idx]['mask']
            coarses = outputs[idx]['coarse']
            completes = outputs[idx]['complete']

            iterator = zip(origins, masks, coarses, completes)

            for origin, mask, coarse, complete in iterator:
                table.add_data(
                        wandb.Image(origin),
                        wandb.Image(mask),
                        wandb.Image(coarse),
                        wandb.Image(complete),
                    )

        current_epoch = self.current_epoch
        metrics = {f'samples_{current_epoch:02d}': table}
        self.logger.log_metrics(metrics)
