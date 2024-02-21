import os
import sys
from typing import Iterator
from easydict import EasyDict
from os.path import dirname as up

from torch import nn, Tensor
from torch.nn import Parameter
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights, ResNet

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.basemodel import Model


class FineTuneResNet(Model):
    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 p_dropout: float,
                 freeze_resnet: bool=True
                 ) -> None:   
        super(FineTuneResNet, self).__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_begin = nn.Sequential(*(list(resnet.children())[:-1]))

        if freeze_resnet:
            for param in self.resnet_begin.parameters():
                param.requires_grad = False
        self.resnet_begin.eval()
        
        self.fc1 = nn.Linear(in_features=512, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        input: x is a tensor of shape (batch_size, 3, 128, 128)
        output: y is a tensor of shape (batch_size, num_classes)
        """
        x = self.resnet_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x
    
    def forward_and_get_intermediare(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        input:
            - x is a tensor of shape (batch_size, 3, 128, 128)
        
        output:
            - intermediare is a tensor of shape (batch_size, 1000)
            - reel_output  is a tensor of shape (batch_size, num_classes) 
        """
        x = self.resnet_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        intermediare = self.relu(self.fc1(x))
        x = self.dropout(intermediare)
        reel_output = self.fc2(x)
        return intermediare, reel_output
    
    def train(self) -> None:
        self.dropout = self.dropout.train()
    
    def eval(self) -> None:
        self.dropout = self.dropout.eval()
    
    def get_intermediare_parameters(self) -> Iterator[Parameter]:
        return self.fc1.parameters()


def get_resnet(config: EasyDict) -> FineTuneResNet:
    """ return resnet according the configuration """
    resnet = FineTuneResNet(num_classes=config.data.num_classes,
                            **config.model.resnet)
    return resnet


def get_original_resnet(finetune_resnet: FineTuneResNet) -> ResNet:
    """ get the orginal resnet with weigth matching """
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    finetune_weigth = finetune_resnet.resnet_begin.state_dict()

    loaded_param : list[str] = []
    num_fc_layers: int = 0
    count: int = 0

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
    import torch
    config_path = 'config/config.yaml'   
    config = EasyDict(yaml.safe_load(open(config_path)))

    model = get_resnet(config)
    print("Total parameters:", model.get_number_parameters())
    print("Trainable parameters:", model.get_number_learnable_parameters())
    learnable_param = model.get_dict_learned_parameters()
    print('learnable parameters', learnable_param)
    model.load_dict_learnable_parameters(state_dict=learnable_param, strict=True)

    x = torch.randn((32, 3, 128, 128))
    y = model.forward(x)

    print("y shape:", y.shape)