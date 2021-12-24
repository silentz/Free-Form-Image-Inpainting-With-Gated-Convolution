import os
import urllib.request
import tarfile
import torchvision

from torch.utils.data import Dataset
from typing import Any, Dict, Callable
from .mask import generate_random_mask


_datasets = {
    'places365': {
        'url': 'http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar',
        'archive': 'places365standard_easyformat.tar',
        'target': 'places365_standard',
    }
}


class Downloadable:

    def __init__(self, root: str,
                       target: str,
                       archive: str,
                       link: str):
        data_path = os.path.join(root, target)
        archive_path = os.path.join(root, archive)

        if not os.path.isdir(data_path):
            urllib.request.urlretrieve(link, archive_path)

            if archive_path.endswith('.tar.gz'):
                tar = tarfile.open(archive_path, 'r:gz')
                tar.extractall(path=root)
                tar.close()


class Places(Dataset, Downloadable):

    def __init__(self, root: str,
                       split: str):
        target =  _datasets['places365']['target']
        archive = _datasets['places365']['archive']
        link =    _datasets['places365']['url']
        super().__init__(root, target, archive, link)

        meta = os.path.join(root, target, f'{split}.txt')
        with open(meta, 'r') as file:
            meta = file.read().strip().split('\n')

        self._root = os.path.join(root, target)
        self._meta = meta

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self._meta[idx]
        filepath = os.path.join(self._root, filename)
        image = torchvision.io.read_image(
                path=filepath,
                mode=torchvision.io.ImageReadMode.RGB,
            )
        result = {'image': image}
        return result


class PlacesWithMask(Places):

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = super().__getitem__(idx)['image']
        height, width = image.shape[1], image.shape[2]
        mask = generate_random_mask(height, width)

        return {
                'image': image,
                'mask': mask,
            }

