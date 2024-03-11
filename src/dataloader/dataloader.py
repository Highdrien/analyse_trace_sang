import os
import sys
from PIL import Image
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.dataloader.transforms import get_transforms
from src.dataloader.labels import LABELS, BACKGROUND


class DataGenerator(Dataset):
    def __init__(self, config: EasyDict, mode: str, use_background: bool) -> None:
        if mode not in ["train", "val", "test"]:
            raise ValueError(
                f"Error, expected mode is train, val, or test but found: {mode}"
            )
        self.mode = mode
        self.use_background = use_background

        dst_path = os.path.join(config.data.path, f"{mode}_{config.data.image_size}")
        print(f"dataloader for {mode}, datapath: {dst_path}")
        if not os.path.exists(dst_path):
            raise FileNotFoundError(
                f"{dst_path} wans't found. ",
                f"Make sure that you have run get_data_transform correctly",
                f"with the image_size={config.data.image_size}",
            )

        self.data: list[tuple[str, str, str]] = []
        for label in LABELS:
            # add background
            if self.use_background:
                for background in BACKGROUND:
                    folder = os.path.join(dst_path, label, background)
                    if not os.path.exists(folder):
                        raise FileNotFoundError(f"{folder} wasn't found")
                    for image_name in os.listdir(folder):
                        self.data.append((os.path.join(folder, image_name), label, background))
            else:
                folder = os.path.join(dst_path, label)
                if not os.path.exists(folder):
                    raise FileNotFoundError(f"{folder} wasn't found")
                for image_name in os.listdir(folder):
                    self.data.append((os.path.join(folder, image_name), label, None))

        self.transform = get_transforms(transforms_config=config.data.transforms,
                                        mode=mode)
        print(self.transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
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

        item: dict[str, Tensor] = {}

        # Get image
        img = Image.open(image_path)
        item['image'] = self.transform(img)

        # Get label
        if label not in LABELS:
            raise ValueError(f"Expected label in LABEL but found {label}")
        item['label'] = torch.tensor(LABELS.index(label), dtype=torch.int64)

        # Get background
        if self.use_background:
            if background not in BACKGROUND:
                raise ValueError(f"Expected background in {BACKGROUND}",
                                 f" but found {background}")
            item['background'] = torch.tensor(BACKGROUND.index(background),
                                              dtype=torch.int64)

        return item


def create_dataloader(config: EasyDict,
                      mode: str,
                      use_background: bool = True
                      ) -> DataLoader:
    generator = DataGenerator(
        config=config,
        mode=mode,
        use_background=use_background
    )

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

    dataloader = create_dataloader(config=config, mode="train", use_background=True)
    print(dataloader.batch_size)

    start_time = time.time()
    item: dict[str, Tensor] = next(iter(dataloader))
    stop_time = time.time()
    x = item['image']
    label = item['label']
    background = item['background']

    print(f"time to load a batch: {stop_time - start_time:2f}s", end='')
    print(f"for a batchsize={config.learning.batch_size}")

    print(x.shape, x.dtype)
    print(label, label.shape, label.dtype)
    print(background, background.shape, background.dtype)

    plot_batch(x=x)
