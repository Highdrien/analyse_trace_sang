import os
import sys
from PIL import Image
from typing import Tuple
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.dataloader.transforms import get_transforms


LABELS = [
    "1- Modèle Traces passives",
    "2- Modèle Goutte à Goutte",
    "3- Modèle Transfert par contact",
    "4- Modèle Transfert glissé",
    "5- Modèle Altération par contact",
    "6- Modèle Altération glissée",
    "7- Modèle d'Accumulation",
    "8- Modèle Coulée",
    "9- Modèle Chute de volume",
    "10- Modèle Sang Propulsé",
    "11- Modèle d'éjection",
    "12- Modèle Volume impacté",
    "13- Modèle Imprégnation",
    "14- Modèle Zone d'interruption",
    "15- Modèle Modèle d'impact",
    "16- Modèle Foyer de modèle d'impact",
    "17- Modèle Trace gravitationnelle",
    "18- Modèle Sang expiré",
]
BACKGROUND = ["carrelage", "papier", "bois", "lino"]


class DataGenerator(Dataset):
    def __init__(self, config: EasyDict, mode: str) -> None:
        if mode not in ["train", "val", "test"]:
            raise ValueError(
                f"Error, expected mode is train, val, or test but found: {mode}"
            )
        self.mode = mode

        dst_path = os.path.join(config.data.path, f"{mode}_{config.data.image_size}")
        print(f"dataloader for {mode}, datapath:{dst_path}")
        if not os.path.exists(dst_path):
            raise FileNotFoundError(
                f"{dst_path} wans't found. Make sure that you have run get_data_transform",
                f"with the image_size={config.data.image_size}",
            )

        self.data: Tuple[str, str, str] = []
        for label in LABELS:
            for background in BACKGROUND:
                folder = os.path.join(dst_path, label, background)
                if not os.path.exists(folder):
                    raise FileNotFoundError(f"{folder} wasn't found")
                for image_name in os.listdir(folder):
                    self.data.append(
                        (os.path.join(folder, image_name), label, background)
                    )

        self.transform = get_transforms(
            transforms_config=config.data.transforms, mode=mode
        )
        print(self.transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        input: index of the image to load
        output: tuple (x, label, background)
        -----
        SHAPE & DTYPE
        x:      (3, image_size, image_size)     torch.float32
        label:  (1)                             torch.int64
        backg:  (1)                             torch.int64
        """
        image_path, label, background = self.data[index]

        # Get image
        img = Image.open(image_path)
        x = self.transform(img)

        # Get label
        if label not in LABELS:
            raise ValueError(f"Expected label in LABEL but found {label}")
        label = torch.tensor(LABELS.index(label), dtype=torch.int64)

        # Get background
        if background not in BACKGROUND:
            raise ValueError(
                f"Expected backdround in BACKGROUND but found {background}"
            )
        background = torch.tensor(BACKGROUND.index(background), dtype=torch.int64)

        return x, label, background


def create_dataloader(config: EasyDict, mode: str) -> DataLoader:
    generator = DataGenerator(config=config, mode=mode)
    dataloader = DataLoader(
        dataset=generator,
        batch_size=config.learning.batch_size,
        shuffle=config.learning.shuffle,
        drop_last=config.learning.drop_last,
        num_workers=config.learning.num_workers,
    )

    return dataloader


if __name__ == "__main__":
    import sys
    import yaml
    import time
    from os.path import dirname as up

    sys.path.append(up(up(up(os.path.abspath(__file__)))))
    from src.dataloader.show_batch import plot_batch

    config_path = "config/config.yaml"
    config = EasyDict(yaml.safe_load(open(config_path)))
    config.learning.num_workers = 1

    # generator = DataGenerator(config=config, mode='train')
    # print(len(generator))
    # x, label, background = generator.__getitem__(index=3)
    # ic(x.shape, x.dtype)
    # ic(label, label.shape, label.dtype)
    # ic(background, background.shape, background.dtype)

    dataloader = create_dataloader(config=config, mode="train")
    print(dataloader.batch_size)

    start_time = time.time()
    x, label, background = next(iter(dataloader))
    stop_time = time.time()
    print(
        f"time to load a batch: {stop_time - start_time:2f}s for a batchsize={config.learning.batch_size}"
    )
    print(x.shape, x.dtype)
    print(label, label.shape, label.dtype)
    print(background, background.shape, background.dtype)

    plot_batch(x=x)
