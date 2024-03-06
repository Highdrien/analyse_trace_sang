import os
import yaml
import copy
import random
from typing import Any
from easydict import EasyDict
from dataclasses import dataclass


class Search:
    def __init__(self,
                 config_yaml_file: str,
                 search_yaml_file: str
                 ) -> None:
        if not os.path.exists(config_yaml_file):
            raise FileNotFoundError(config_yaml_file)
        if not os.path.exists(search_yaml_file):
            raise FileNotFoundError(search_yaml_file)
        
        self.config = self.loadyaml(config_yaml_file)
        search = self.loadyaml(search_yaml_file)

        self.items: list[Item] = []
        self.get_all_item(search)
        if self.items == []:
            raise ValueError(f'Empty items')

        print('\n---- ITEMS POSSIBILITIES ----')
        for item in self.items:
            print(item)
        print('')

        self.len: int = 1
        for item in self.items:
            self.len *= len(item)
        print(f'{self.len} possibilities')
    
    def loadyaml(self, yaml_file: str) -> EasyDict:
        return EasyDict(yaml.safe_load(open(yaml_file, 'r')))
    
    def get_all_item(self, search: EasyDict, keys: list[str] = []) -> None:
        for key, value in search.items():
            new_keys: list[str] = keys + [key]
            if type(value) == EasyDict:
                self.get_all_item(value, new_keys)
            elif type(value) == list:
                self.items.append(Item(key=new_keys, possibles_values=value))
    
    def __len__(self) -> int:
        return self.len
    

class RandomSearch(Search):
    def __init__(self, config_yaml_file: str, search_yaml_file: str) -> None:
        super().__init__(config_yaml_file, search_yaml_file)
    
    def copy_and_change_config(self, search: EasyDict, keys: list[str] = []) -> None:
        for key, value in search.items():
            new_keys: list[str] = keys + [key]
            if type(value) == EasyDict:
                self.get_all_item(value, new_keys)
            elif type(value) == list:
                self.items.append(Item(key=new_keys, possibles_values=value))
        
        


    
@dataclass
class Item:
    key: list[str]                  # key path: config[key[0]][key[1]]... to get the item
    possibles_values: list[Any]     # possibles values for the item
    
    def get_value(self, index_value: int = None) -> Any:
        """ get a index_value of possibles_value 
        if index_value is None -> take an random possibles value"""

        if not (0 < index_value < len(self)):
            raise ValueError('index_value out of range. '
                             f'{index_value = } but {len(self) = }')

        if index_value is None:
            index_value = random.randint(0, len(self) -1)
        
        return self.possibles_values[index_value]
    
    def __len__(self) -> int:
        return len(self.possibles_values)
    
    def __repr__(self) -> str:
        return f'{str(self.key):<50} : {self.possibles_values}'


if __name__ == '__main__':
    search = Search(config_yaml_file=os.path.join('config', 'config.yaml'),
                    search_yaml_file=os.path.join('config', 'search.yaml'))

    