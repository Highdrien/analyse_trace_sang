import os
import sys
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.labels import get_topk_prediction
from src.dataloader.infer_dataloader import create_infer_dataloader
from src.model import finetune_resnet
from utils import utils


def infer(infer_dataloader: DataLoader,
          infer_datapath: str,
          logging_path: str,
          config: EasyDict,
          dstpath: str,
          filename: str,
          run_temperature_optimization: bool = True,
          sep: str = ','
          ) -> None:
    """
    Perform inference using the provided dataloader and model.

    Args:
        infer_dataloader (DataLoader): The dataloader for inference (can be None if infer_datapath is specify).
        infer_datapath (str): The path to the inference data (only in the case of infer_dataloader is None).
        logging_path (str): The path to the logging directory.
        config (EasyDict): The configuration object (most of times, in the logging_path).
        dstpath (str): The destination path for saving the inference results.
        filename (str): The filename for saving the inference results.
        run_temperature_optimization (bool, optional): Whether to run temperature optimization. Defaults to True.
        sep (str, optional): The separator for saving the inference results. Defaults to ','.

    Raises:
        ValueError: If both infer_dataloader and infer_datapath are None.
    """
    device = utils.get_device(device_config=config.learning.device)

    if infer_dataloader is None:
        if infer_datapath is not None:
            infer_dataloader = create_infer_dataloader(config=config,
                                                       datapath=infer_datapath)
        else:
            raise ValueError('infer_generator and infer_datapath cannot be both None')

    # Get model
    model = finetune_resnet.get_finetuneresnet(config)
    weight = utils.load_weights(logging_path, device=device, model_name='res')
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    model = model.to(device)
    del weight

    temperature: float = 1.5
    output: list[list[tuple[int, str, float]]] = []
    images_paths: list[str] = []

    model.eval()
    with torch.no_grad():
        for x, image_path in infer_dataloader:
            x: Tensor = x.to(device)
            y_pred = model.forward(x)

            if run_temperature_optimization:
                y_pred = torch.nn.functional.softmax(y_pred / temperature, dim=-1)
            else:
                y_pred = torch.nn.functional.softmax(y_pred, dim=-1)

            output += get_topk_prediction(y_pred, k=3)
            images_paths += list(image_path)
        
    save_infer(dstpath=dstpath,
               filename=filename,
               output=output,
               images_paths=images_paths,
               sep=sep)
    

def save_infer(dstpath: str,
               filename: str,
               output: list[list[tuple[int, str, float]]],
               images_paths: list[str],
               sep=','
               ) -> None:
    """
    Save the inference results to a file.

    Args:
        dstpath (str): The destination path where the file will be saved.
        filename (str): The name of the file.
        output (list[list[tuple[int, str, float]]]): The inference results.
        images_paths (list[str]): The paths of the input images.
        sep (str, optional): The separator used in the file. Defaults to ','.
    """
    file = os.path.join(dstpath, filename)
    k = len(output[0])

    header = f'Image{sep}'
    for j in range(k):
        header += f'Prediction {j + 1}{sep}Confidence {j + 1} (en %){sep}'
    print(header)

    with open(file, 'w', encoding='utf8') as f:
        f.write(header[:-len(sep)] + '\n')

        for i in range(len(output)):
            line = f'{images_paths[i]}{sep}'
            for j in range(k):
                line += f'{output[i][j][1]}{sep}{output[i][j][2] * 100:.1f}{sep}'
            f.write(line[:-len(sep)] + '\n')
        f.close()

    print(f'Inference results saved at {file}')



if __name__ == '__main__':
    import yaml

    logging_path = os.path.join('logs', 'resnet_img256_0')
    datapath = os.path.join('data', 'images_to_predict')
    config = EasyDict(yaml.safe_load(open(os.path.join(logging_path, 'config.yaml'))))
    
    infer(infer_dataloader=None,
          infer_datapath=datapath,
          logging_path=logging_path,
          config=config,
          run_temperature_optimization=False)
        