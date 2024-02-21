import os
import sys
import yaml
import numpy as np
from easydict import EasyDict
import matplotlib.pyplot as plt
from os.path import dirname as up

from torch import Tensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.resnet import Resnet


def get_saliency_map(model: Resnet,
                     image: np.ndarray | Tensor,
                     plot_map: bool=False,
                     return_label: bool=False
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
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Convert numpy image to torch tensor if necessary
    # if isinstance(image, np.ndarray):
    #     image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).float()

    # Forward pass
    output = model.forward(image)

    # Identify the target layer
    target_layer = model.resnet_begin[7][1].conv2

    # Create GradCAM object
    cam = GradCAM(model=model.true_resnet, target_layers=[target_layer])

    # Define the target category
    targets = None  # None will return the gradients for the highest scoring category.

    # Get the grayscale cam
    grayscale_cam = cam(input_tensor=image, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    # Convert your input tensor to RGB image
    rgb_img = image[0].permute(1, 2, 0).numpy()
    rgb_img = rgb_img.astype(np.float32)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

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
    config_path = 'config/config.yaml'   
    config = EasyDict(yaml.safe_load(open(config_path)))

    from src.model.resnet import get_resnet
    from utils.get_random_image import get_random_img

    model = get_resnet(config)

    #get an image:
    x, label = get_random_img(image_type='torch')
    x: Tensor = x.unsqueeze(dim=0)
    print("x shape:", x.shape)
    print('y_true:', label)

    #get the saliency map
    saliency_map = get_saliency_map(model, x, plot_map=True)
    print("saliency_map shape:", saliency_map.shape)