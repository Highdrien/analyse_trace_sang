import os
import yaml
from easydict import EasyDict


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
        self.search = self.loadyaml(search_yaml_file)

        print(self.config)
        print(self.search)
    
    def loadyaml(self, yaml_file: str) -> EasyDict:
        return EasyDict(yaml.safe_load(open(yaml_file, 'r')))



if __name__ == '__main__':
    search = Search(config_yaml_file=os.path.join('config', 'config.yaml'),
                    search_yaml_file=os.path.join('config', 'search.yaml'))
    