import os
from PIL import Image
from easydict import EasyDict
from typing import List, Tuple

from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class InferDataGenerator(Dataset):
    def __init__(self, datapath: str, image_size: int) -> None:
        """
        # Arguments:
        datapath: int
            can be a folder or a filepath
        image_size: int
            size of the image
        """

        self.data: List[str] = []

        if not datapath.endswith(("jpeg", "png", "jpg")):
            for dirpath, _, filenames in os.walk(datapath):
                if filenames != []:
                    good_files = filter(
                        lambda x: x.endswith(("jpeg", "png", "jpg")), filenames
                    )
                    self.data += list(
                        map(lambda file: os.path.join(dirpath, file), good_files)
                    )
        else:
            self.data = [datapath]

        self.image_size = (image_size, image_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        """
        # Arguments:
        index of the image to load
        # Outputs:
        tensor x with shape (3, image_size, image_size) and dtype torch.float32
        image_path: str path to the image
        """
        image_path = self.data[index]
        image = Image.open(image_path)
        x = self.transform(image)
        return x, image_path


def create_dataloader(config: EasyDict, datapath: str) -> DataLoader:
    generator = InferDataGenerator(datapath, image_size=config.data.image_size)
    dataloader = DataLoader(
        dataset=generator,
        batch_size=config.learning.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.learning.num_workers,
    )

    return dataloader


if __name__ == "__main__":
    datapath = r"data\data_labo\test_512"
    infer_generator = InferDataGenerator(datapath=datapath, image_size=512)
    print(f"{len(infer_generator)=}")
    x, y = infer_generator.__getitem__(index=0)
    print(f"{x.shape=}")
