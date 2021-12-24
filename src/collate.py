import torch
from dataclasses import dataclass


@dataclass
class Batch:
    images: torch.Tensor
    masks: torch.Tensor

    def to(self, device: torch.device) -> 'Batch':
        return Batch(
                images=self.images.to(device),
                masks=self.masks.to(device),
            )


def collate_fn(inputs: list) -> Batch:
    images = [x['image'] for x in inputs]
    masks  = [x['mask'] for x in inputs]

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    return Batch(
            images=images,
            masks=masks,
        )
