from typing import Iterator

import torch
from torch import nn
from torch.nn import Parameter


class Model(nn.Module):
    """ BaseModel with method for get parameter more easily """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def get_parameters(self) -> Iterator[Parameter]:
        """ get parameters """
        return self.parameters()
    
    def get_learned_parameters(self) -> Iterator[Parameter]:
        """ get learned parameters """
        for param in self.parameters():
            if param.requires_grad:
                yield param
    
    def get_name_learned_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """ get name and parameters st param is leanable"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def get_number_parameters(self) -> int:
        """ get the number of parameters """
        return sum(p.numel() for p in self.parameters())

    def get_number_learnable_parameters(self) -> int:
        """ get the number of learned parameters """
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 
    
    def get_dict_learned_parameters(self) -> dict[str, Parameter]:
        """ get learned parameters in a dict to save it """
        state_dict: dict[str, Parameter] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state_dict[name] = param

        return state_dict
    
    def load_dict_learnable_parameters(self,
                                       state_dict: dict[str, Parameter],
                                       strict: bool=True,
                                       verbose: bool=False
                                       ) -> None:
        """ load only learnable param from a dict
        # Arguments:
        state_dict: dict[str, Parameter]
            dict which contains the name and the parameters
        strict: bool, default=True
            if True, raise Error if there are some matching error
        verbose: bool, default=True
            if True and strict=False, print Warning Error if there are some matching error
        """
        loaded_keys = []
        for name, param in self.named_parameters():
            # if name in keys
            if name in state_dict.keys():
                if param.requires_grad:
                    with torch.no_grad():
                        param.copy_(state_dict[name])
                    loaded_keys.append(name)
                elif strict or verbose:
                    error_message = f'parameter:{name} is in state_dict.keys ' + \
                                    f'but this param has requires_grad=False'
                    print_error_message(error_message, strict, verbose)
            
            # if name not in keys but param is requires_grad
            elif param.requires_grad:
                error_message = f'parameter:{name} not in state_dict.keys ' + \
                                f'but this param has requires_grad=True'
                print_error_message(error_message, strict, verbose)

        # if loaded_keys != state_dict.keys()   (we know that: state_dict.keys() c loaded_keys)
        if len(loaded_keys) != len(state_dict):
            missing_keys = [key for key in state_dict.keys() if key not in loaded_keys]
            error_message = f"the following parameter wasn't load: {missing_keys}"
            print_error_message(error_message, strict, verbose)
        
        print('dict was successful load')
    


def print_error_message(error_message: str, strict: bool, verbose: bool) -> None:
    if strict:
        raise KeyError(error_message)
    if verbose:
        print(Warning(error_message))
