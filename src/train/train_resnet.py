import os
import sys
import time
from tqdm import tqdm
from easydict import EasyDict
from os.path import dirname as up

import torch

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from config.config import train_step_logger, train_logger
from src.dataloader.dataloader import create_dataloader
from src.metrics import Metrics
from src.model import resnet
from utils import utils, plot_learning_curves


def train(config: EasyDict) -> None:
    if config.model.name != 'resnet':
        raise ValueError(f"Expected model.name=resnet but found {config.model.name}.")
    
    device = utils.get_device(device_config=config.learning.device)

    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    val_generator = create_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator) 
    print(f"Found {n_train} training batches and {n_val} validation batches")

    # Get model
    model = resnet.get_resnet(config)
    model = model.to(device)

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.get_learned_parameters(), lr=config.learning.learning_rate_resnet)

    # Get metrics
    metrics = Metrics(num_classes=config.data.num_classes, run_argmax_on_y_true=False)
    metrics.to(device)

    # Save experiment
    save_experiment = config.learning.save_experiment
    print(f'{save_experiment = }')
    if save_experiment:
        logging_path = train_logger(config, metrics_name=metrics.get_names())
        best_val_loss = 10e6


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        print("epoch: ", epoch)
        train_loss = 0
        train_metrics = metrics.init_metrics()
        train_range = tqdm(train_generator)

        # Training
        model.train()
        for i, (x, y_true, _) in enumerate(train_range):
            x = x.to(device)            # x shape: torch.Size([32, 3, 128, 128])
            y_true = y_true.to(device)  # y_true shape: torch.Size([32])
            y_pred = model.forward(x)   # y_pred shape: torch.Size([32, 2])
            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_metrics += metrics.compute(y_pred, y_true)

            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()

        train_loss = train_loss / n_train
        train_metrics = train_metrics / n_train
        print(metrics.get_info(metrics_value=train_metrics))

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_metrics = metrics.init_metrics()
        val_range = tqdm(val_generator)

        model.eval()

        with torch.no_grad():
            
            for i, (x, y_true, _) in enumerate(val_range):
                x = x.to(device)
                y_true = y_true.to(device)

                y_pred = model.forward(x)

                loss = criterion(y_pred, y_true)

                val_loss += loss.item()
                val_metrics += metrics.compute(y_pred, y_true)

                current_loss = val_loss / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f}")
                val_range.refresh()

        val_loss = val_loss / n_val
        val_metrics = val_metrics / n_val   
        print(metrics.get_info(metrics_value=val_metrics))

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################

        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss,
                              train_metrics=train_metrics,
                              val_metrics=val_metrics)
            
            if val_loss < best_val_loss:
                print('save model weights')
                torch.save(model.get_dict_learned_parameters(),
                           os.path.join(logging_path, 'checkpoint.pt'))
                best_val_loss = val_loss

            #print(f'{best_val_loss = }')

    stop_time = time.time()
    print(f"training time: {stop_time - start_time} secondes for {config.learning.epochs} epochs")

    if save_experiment:
        plot_learning_curves.save_learning_curves(path=logging_path)


if __name__ == '__main__':
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    train(config=config)