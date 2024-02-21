import os
import random
import numpy as np
from PIL import Image
from typing import Literal

from torch import Tensor
from torchvision import transforms


def get_random_img(data_path: str='data/data_labo/test_256',
                   image_type: Literal['numpy', 'torch']='torch'
                   ) -> tuple[np.ndarray | Tensor, str]:
    
    label = random.choice(os.listdir(data_path))
    background = random.choice(os.listdir(os.path.join(data_path, label)))
    folder_path = os.path.join(data_path, label, background)
    images = os.listdir(folder_path)

    if len(images) == 0:
        # if there are not image, find an other one
        return get_random_img(data_path)
    
    image_path = os.path.join(folder_path, random.choice(images))
    print(f'{image_path=}')
    image = Image.open(image_path)

    if image_type == 'numpy':
        image = np.array(image)
    if image_type == 'torch':
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

    return image, label