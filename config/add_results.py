import os
import yaml
from easydict import EasyDict

hyperparameters: dict[str, list[str]] = {
        'image_size': ['data', 'image_size'],
        'brightness': ['data', 'transforms', 'color', 'brightness'],
        'contrast': ['data', 'transforms', 'color', 'contrast'],
        'hidden_size': ['model', 'resnet', 'hidden_size'],
        'p_dropout': ['model', 'resnet', 'p_dropout'],
        'epochs': ['learning', 'epochs'],
        'batch_size': ['learning', 'batch_size'],
        'learning_rate_resnet': ['learning', 'learning_rate_resnet'],
    }


def create_csv_with_all_test_metrics(csv_output: str='hyperparameters_storage.csv',
                                     logs_path: str='logs',
                                     test_file_name: str= 'test_log.txt',
                                     config_name: str= 'config.yaml'
                                     ) -> None:
    logs_with_test = get_logs_with_test(logs_path, test_file_name)

    metrics_name = ['acc micro', 'acc macro']
    hyperparameters: dict[str, list[str]] = {
        'image_size': ['data', 'image_size'],
        'brightness': ['data', 'transforms', 'color', 'brightness'],
        'contrast': ['data', 'transforms', 'color', 'contrast'],
        'hidden_size': ['model', 'resnet', 'hidden_size'],
        'p_dropout': ['model', 'resnet', 'p_dropout'],
        'epochs': ['learning', 'epochs'],
        'batch_size': ['learning', 'batch_size'],
        'learning_rate_resnet': ['learning', 'learning_rate_resnet'],
    }

    all_test_results: str = 'logs,' + list_into_str(metrics_name) + ',' + \
                            list_into_str(hyperparameters.keys()) + '\n'

    for log in logs_with_test:
        log_name = log.split(os.sep)[-1]
        test_results = get_test_results(log, test_file_name)
        config = get_config(log, hyperparameters, config_name)
        test_metrics = list(map(lambda metric_name: test_results[metric_name], metrics_name))
        all_test_results += f'{log_name},{list_into_str(test_metrics, round_up=True)},{list_into_str(config)}\n'
    
    with open(file=os.path.join(logs_path, csv_output), mode='w', encoding='utf8') as f:
        f.write(all_test_results)
        f.close()

        

def get_test_results(log_path: str,
                     test_file_name: str='test_log.txt'
                     ) -> dict[str, float]:
    test_file = os.path.join(log_path, test_file_name)
    if not os.path.exists(path=test_file):
        raise FileNotFoundError(f"{test_file} wasn't found")
    
    test_results: dict[str, float] = {}
    with open(test_file, mode='r', encoding='utf8') as f:
        for line in f.readlines():
            try:
                metrics_name, metric_value = line[:-1].split(': ')
            except:
                raise ValueError(f'Can split the line{line[-1]} with ": "')
            
            test_results[metrics_name] = float(metric_value)
    
    return test_results


def get_config(log_path: str,
               hyperparameters: dict[str, list[str]],
               config_name: str='config.yaml'
               ) -> list:
    config_path = os.path.join(log_path, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config path was not found in {config_path}')
    
    config = yaml.safe_load(open(config_path))

    output = []
    for keys in hyperparameters.values():
        value = config
        for key in keys:
            value = value[key]
        output.append(value)
    
    return output


def get_logs_with_test(logs_path: str='logs',
                       test_file_name: str='test_log.txt'
                       ) -> list[str]:
    logs = filter(lambda x: '.' not in x, os.listdir(logs_path))
    logs = map(lambda x: os.path.join(logs_path, x), logs)

    logs_with_test = filter(lambda x: test_file_name in os.listdir(x), logs)

    return list(logs_with_test)


def list_into_str(l: list, sep: str=',', round_up: bool=False) -> str:
    output: str = ''
    for x in l:
        if round_up and type(x) == float:
            output += f'{x:.3f}{sep}'
        else:
            output += f'{x}{sep}'
    return output[:-len(sep)]


if __name__ == '__main__':
    test_file = os.path.join('logs', 'resnet_0', 'test_log.txt')
    create_csv_with_all_test_metrics()