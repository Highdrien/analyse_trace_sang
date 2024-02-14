import os
import sys
import time
import numpy as np
from tqdm import tqdm
from icecream import ic
from itertools import chain
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.optim import Adam

# sys.path.append(up(os.path.abspath(__file__)))
# sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))

from config.config import train_step_logger, train_logger
from src.dataloader.dataloader import create_dataloader
from src.metrics import Metrics
from src.model import resnet, adversarial
from utils import utils, plot_learning_curves


def train(config: EasyDict) -> None:
    if config.model.name != 'adversarial':
        raise ValueError(f"Expected model.name=adversarial but found {config.model.name}.")

    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    val_generator = create_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator) 
    print(f"Found {n_train} training batches and {n_val} validation batches")

    # Get model
    res_model = resnet.get_resnet(config)
    adv_model = adversarial.get_adv(config)

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    alpha = config.learning.alpha # weight of the adversarial loss

    # Optimizer and Scheduler
    resnet_optimizer = Adam(res_model.get_learned_parameters(),
                            lr=config.learning.learning_rate_resnet)
    adv_optimizer = Adam(chain(adv_model.parameters(), res_model.get_intermediare_parameters()),
                         lr=config.learning.learning_rate_adversary)

    # Get metrics
    res_metrics = Metrics(num_classes=config.data.num_classes, run_argmax_on_y_true=False)
    adv_metrics = Metrics(num_classes=4, run_argmax_on_y_true=False)
    metrics_name = ['res loss', 'adv loss'] + utils.get_metrics_name_for_adv(res_metrics, adv_metrics)

    # Get and put on device
    device = utils.get_device(device_config=config.learning.device)
    utils.put_on_device(device, res_model, adv_model, res_metrics, adv_metrics)

    # Save experiment
    save_experiment = config.learning.save_experiment
    print(f'{save_experiment = }')
    if save_experiment:
        logging_path = train_logger(config, metrics_name=metrics_name)
        best_val_loss = 10e6


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        print("epoch: ", epoch)
        train_loss = 0
        train_metrics = np.zeros(len(metrics_name))
        train_range = tqdm(train_generator)

        # Training
        res_model.train()
        adv_model.train()
        for i, (x, res_true, adv_true) in enumerate(train_range):
            x: Tensor = x.to(device)
            res_true: Tensor = res_true.to(device)
            adv_true: Tensor = adv_true.to(device)

            inter, res_pred = res_model.forward_and_get_intermediare(x)
            adv_pred = adv_model.forward(x=inter)

            res_loss = criterion(res_pred, res_true)
            adv_loss = criterion(adv_pred, adv_true)

            crossloss = res_loss - alpha * adv_loss

            adv_loss.backward(retain_graph=True)
            crossloss.backward()

            adv_optimizer.step()
            resnet_optimizer.step()

            resnet_optimizer.zero_grad()
            adv_optimizer.zero_grad()

            train_loss += crossloss.item()
            train_metrics += np.concatenate((np.array([res_loss.item(), adv_loss.item()]),
                                             res_metrics.compute(y_pred=res_pred, y_true=res_true),
                                             adv_metrics.compute(y_pred=adv_pred, y_true=adv_true)))

            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_metrics = np.zeros(len(metrics_name))
        val_range = tqdm(val_generator)

        res_model.eval()
        adv_model.eval()
        with torch.no_grad():
            
            for i, (x, res_true, adv_true) in enumerate(val_range):
                x: Tensor = x.to(device)
                res_true: Tensor = res_true.to(device)
                adv_true: Tensor = adv_true.to(device)

                inter, res_pred = res_model.forward_and_get_intermediare(x)
                adv_pred = adv_model.forward(x=inter)

                res_loss = criterion(res_pred, res_true)
                adv_loss = criterion(adv_pred, adv_true)

                crossloss = res_loss - alpha * adv_loss

                val_loss += crossloss.item()
                val_metrics += np.concatenate((np.array([res_loss.item(), adv_loss.item()]),
                                               res_metrics.compute(y_pred=res_pred, y_true=res_true),
                                               adv_metrics.compute(y_pred=adv_pred, y_true=adv_true)))

                current_loss = val_loss / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f}")
                val_range.refresh()
          

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        train_metrics = train_metrics / n_train
        val_metrics = val_metrics / n_val

        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss,
                              train_metrics=train_metrics,
                              val_metrics=val_metrics)
            
            if val_loss < best_val_loss:
                print('save model weights')
                torch.save(res_model.get_dict_learned_parameters(),
                           os.path.join(logging_path, 'res_checkpoint.pt'))
                torch.save(adv_model.get_dict_learned_parameters(),
                           os.path.join(logging_path, 'adv_checkpoint.pt'))
                best_val_loss = val_loss

            print(f'{best_val_loss = }')

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")
    
    if save_experiment:
        plot_learning_curves.save_learning_curves(path=logging_path)


if __name__ == '__main__':
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    config.model.name = 'adversarial' 
    train(config=config)