import os
import sys
from os.path import dirname as up

import torch
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights, ResNet

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.finetune_resnet import FineTuneResNet


def get_original_resnet(finetune_resnet: FineTuneResNet) -> ResNet:
    """ get the orginal resnet with weigth matching """
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    finetune_weigth = finetune_resnet.resnet_begin.state_dict()

    loaded_param : list[str] = []
    num_fc_layers: int = 0

    for name, param in resnet.named_parameters():
        if 'fc' not in name:
            if 'layer' in name:
                new_name = name[len('layer'):]
                finetune_name = str(int(new_name[0]) + 3) + new_name[1:]
            elif name in ['conv1.weight', 'bn1.weight', 'bn1.bias']:
                replace = lambda x: '0' if x == 'conv1' else '1'
                name_split = name.split('.')
                finetune_name = replace(name_split[0]) + '.' + name_split[1]
            else:
                raise ValueError(f'not matched name param: {name}')

            if finetune_name not in finetune_weigth.keys():
                raise ValueError(f'param {name} was not found in finetune_weigth')
            
            with torch.no_grad():
                param.copy_(finetune_weigth[finetune_name])
            loaded_param.append(finetune_name)
        
        else:
            num_fc_layers += 1
    
    return resnet


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from src.model.finetune_resnet import get_finetuneresnet

    # config_path = 'config/config.yaml'
    config_path = os.path.join('logs', 'resnet_2', 'config.yaml')
    config = EasyDict(yaml.safe_load(open(config_path)))

    finetune_resnet = get_finetuneresnet(config)
    print(finetune_resnet)
    resnet = get_original_resnet(finetune_resnet)
    print(resnet)

    x = torch.randn((32, 3, 128, 128))
    y = resnet.forward(x)

    print("y shape:", y.shape)