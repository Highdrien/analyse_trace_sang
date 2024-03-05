import os
import sys
import random
import numpy as np
from PIL import Image
from easydict import EasyDict
from typing import Any, Literal
from os.path import dirname as up

import torch
from torch import Tensor
from torch.nn import Parameter
from torchvision import transforms

sys.path.append(up(os.path.abspath(__file__)))

from src.metrics import Metrics


def get_device(device_config: str) -> torch.device:
    """ get device: cuda or cpu """
    if torch.cuda.is_available() and device_config == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def put_on_device(device: torch.device, *args: Any) -> None:
    """ put all on device """
    for arg in args:
        arg = arg.to(device)


def get_metrics_name_for_adv(resnet_metrics: Metrics,
                             adv_metrics: Metrics
                             ) -> list[str]:
    """ get all metrics name """
    add_name = lambda model_name, metric_name: f'{model_name}_{metric_name}'
    add_resnet = lambda metric_name: add_name(model_name='resnet', metric_name=metric_name)
    add_adv = lambda metric_name: add_name(model_name='adv', metric_name=metric_name)
    metrics_name = list(map(add_resnet, resnet_metrics.get_names())) + \
                   list(map(add_adv, adv_metrics.get_names()))
    return metrics_name


def load_weights(logging_path: str,
                 model_name: str = 'res',
                 device: torch.device = torch.device('cpu'),
                 endfile: str = '.pt'
                 ) -> dict[str, Parameter]:
    """
    Load weights from the specified logging path for a given model.

    Args:
        logging_path (str):
            The path where the weight files are stored.
        model_name (str, optional):
            The name of the model. Defaults to 'res'.
        device (torch.device, optional):
            The device to load the weights onto. Defaults to torch.device('cpu').
        endfile (str, optional):
            The file extension of the weight files. Defaults to '.pt'.

    Returns:
        dict[str, Parameter]:
            A dictionary containing the loaded weights.

    """
    weight_files = list(filter(lambda x: x.endswith(endfile),
                               os.listdir(logging_path)))

    if len(weight_files) == 1:
        weight = torch.load(os.path.join(logging_path, weight_files[0]),
                            map_location=device)
        return weight

    model_name_files = list(filter(lambda x: model_name in x,
                                   os.listdir(logging_path)))
    if len(model_name_files) > 1:
        raise FileExistsError(f'Confused by multiple weights for the {model_name} model')
    if len(model_name_files) < 1:
        raise FileNotFoundError(f'No weights was found in {logging_path}')
    weight = torch.load(os.path.join(logging_path, model_name_files[0]),
                        map_location=device)
    return weight


def get_random_img(data_path: str = 'data/data_labo/test_256',
                   image_type: Literal['numpy', 'torch'] = 'torch'
                   ) -> tuple[np.ndarray | Tensor, str]:
    """ get a random image from the test dataset """
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


def convert_tensor_to_rgb(image: Tensor, normelize: bool = False) -> np.ndarray[float]:
    """ convert a torch tensor into a numpy array
    run a permuation in order to have an array with a shape like (256, 256, 3)"""
    rgb_img: np.ndarray = image.permute(1, 2, 0).numpy()
    rgb_img = rgb_img.astype(np.float32)
    if normelize:
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    return rgb_img


def resume_training(config: EasyDict, model: torch.nn.Module) -> None:
    if 'resume_training' not in config.model.resnet.keys():
        print("didn't find resume_training key")
        return None

    resume_training: EasyDict = config.model.resnet.resume_training
    if not resume_training.do_resume:
        print("resume training: None")
        return None

    print(f'resume training from {resume_training.path}')
    weight = load_weights(resume_training.path)
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    del weight

    if not resume_training.freeze_param:
        print('unfreezing the parameters')
        for param in model.parameters():
            param.requires_grad = True

    return None
