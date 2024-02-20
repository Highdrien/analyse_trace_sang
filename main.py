import os
import argparse

from config.config import load_config, find_config
# from src.train import train_resnet, train_adversarial, train_segmentator_v2
from src.train import train_resnet
from src import test, infer


MODE_IMPLEMENTED = ['train', 'test', 'infer']
# MODEL_IMPLEMENTED = ['resnet', 'unet', 'adversarial']
MODEL_IMPLEMENTED = ['resnet']


def main(options: dict) -> None:

    # TRAINING
    if options['mode'] == 'train':
        config = load_config(options['config_path'])

        if config.model.name not in MODEL_IMPLEMENTED:
            raise ValueError(f'Expected model name in {MODEL_IMPLEMENTED} but found {config.model.name}.')
        print(f'train {config.model.name}')

        if config.model.name == 'resnet':
            train_resnet.train(config)
        # if config.model.name == 'adversarial':
        #     train_adversarial.train(config)
        # if config.model.name == 'unet':
        #     train_segmentator_v2.train(config)
    
    # TESTING
    if options['mode'] == 'test':
        if options['path'] is None:
            raise ValueError('Please specify the path to the experiments')
        config = load_config(find_config(experiment_path=options['path']))
        if config.model.name != 'resnet':
            print(f'Attention pas sur que le test va marcher car il est addapter pour resnet et pas pour {config.model.name}.')
        test.test(config=config, logging_path=options['path'])
    
    # INEFRENCE
    if options['mode'] == 'infer':
        if options['path'] is None:
            raise ValueError('Please specify the path to the experiments')
        config = load_config(find_config(experiment_path=options['path']))
        infer.infer(datapath=r'data\data_labo\test_512',
                    logging_path=options['path'],
                    config=config,
                    test_inference=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('mode', default=None, type=str,
                        choices=['train', 'test'])
    parser.add_argument('--config_path', '-c', default=os.path.join('config', 'config.yaml'),
                        type=str, help="path to config (for training)")
    parser.add_argument('--path', '-p', type=str,
                        help="experiment path (for test)")
    parser.add_argument('--inferpath', '-i', type=str,
                        help="experiment path (for test)")
    args = parser.parse_args()
    options = vars(args)

    main(options)