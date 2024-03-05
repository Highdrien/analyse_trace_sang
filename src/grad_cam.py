import os
import sys
import yaml
import numpy as np
from easydict import EasyDict
import matplotlib.pyplot as plt
from os.path import dirname as up

import torch
from torch import Tensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(up(up(os.path.abspath(__file__))))

from src.model.finetune_resnet import FineTuneResNet
from src.model.resnet import get_original_resnet
from utils import utils


def get_saliency_map(model: FineTuneResNet,
                     image: np.ndarray | Tensor,
                     plot_map: bool = False,
                     return_label: bool = False
                     ) -> np.ndarray | tuple[np.ndarray, Tensor]:
    """
    Generates a saliency map for the given image using the provided model.

    Args:
        model: (torch.nn.Module): The model used for generating the saliency map.
        image: (numpy.ndarray or torch.Tensor): The input image for which the saliency map is generated.
            If `image` is a numpy array, it should have shape (height, width, channels).
            If `image` is a torch tensor, it should have shape (batch_size, channels, height, width).
        plot_map (bool, optional): Whether to plot and display the saliency map. Default is False.
        return_label (bool, optional): Whether to return the output label along with the saliency map. Default is False.

    Returns:
        numpy.ndarray: The saliency map as a numpy array with shape (height, width, channels).
        If `return_label` is True, the output label is also returned as a torch.Tensor.

    """
    output = model.forward(image)
    print(f"prédiction du resnet: {torch.argmax(output[0]).item()}")

    true_resnet = get_original_resnet(model) # les parametres sont tous apprenable

    target_layer = true_resnet.layer4[1].conv2
    print(target_layer)

    cam = GradCAM(model=true_resnet, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image, targets=None)

    grayscale_cam = grayscale_cam[0, :]

    rgb_img = utils.convert_tensor_to_rgb(image=image[0], normelize=True)

    visualization = show_cam_on_image(rgb_img, grayscale_cam)

    if plot_map:
        plt.figure(figsize=(10, 10))
        plt.imshow(visualization)
        plt.show()

    if return_label:
        return visualization, output
    else:
        return visualization


if __name__ == '__main__':
    # config_path = 'config/config.yaml'  
    config_path = os.path.join('logs', 'resnet_7') 
    config = EasyDict(yaml.safe_load(open(os.path.join(config_path, 'config.yaml'))))

    from src.model.finetune_resnet import get_finetuneresnet
    from utils import utils

    # config_path = os.path.join('logs', 'resnet_2')
    model = get_finetuneresnet(config)
    weight = utils.load_weights(config_path, device=torch.device('cpu'))
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    del weight

    x, label = utils.get_random_img(image_type='torch')
    x: Tensor = x.unsqueeze(dim=0)
    print("x shape:", x.shape)
    print('y_true:', label)

    saliency_map = get_saliency_map(model, image=x, plot_map=True)
    print("saliency_map shape:", saliency_map.shape)