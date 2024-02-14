import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from os.path import dirname as up

import torch
from torch.optim import Adam

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.explainable.create_mask import batched_segmentation
from src.dataloader.dataloader import create_dataloader
from src.model.segmentation_model import UNet, Classifier
from utils import utils


def train(config: EasyDict, unet: UNet, classifier: Classifier, k: float, pretrain_epochs: int,train_epochs: int,fine_tuning: bool = True) -> None:
    
    if config.model.name != 'unet':
        raise ValueError(f"Expected model.name=unet but found {config.model.name}.")
    
    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    val_generator = create_dataloader(config=config, mode='val')
    test_dataloader = create_dataloader(config=config, mode='test')
    n_train, n_val = len(train_generator), len(val_generator) 
    print(f"Found {n_train} training batches and {n_val} validation batches")

    # Loss
    criterion_unet = torch.nn.BCEWithLogitsLoss()
    criterion_classif = torch.nn.CrossEntropyLoss()

    # Optimizer and Scheduler
    unet_optimizer = Adam(unet.parameters(), lr=0.001)
    classifier_optimizer = Adam(classifier.parameters(), lr=0.001)

    #number of parameters for both 
    print("nb_parameters of unet:",unet.count_parameters())
    print("nb_parameters of classif:",classifier.count_parameters())
    

    # Get and put on device
    device = utils.get_device(device_config=config.learning.device)
    unet.to(device) 
    classifier.to(device)


    # Pretrain unet on k% of the data
    pretrain_batches = int(k * n_train)
    unet.train()

    # Lists to store losses
    pretrain_losses = []
    train_losses = []
    val_losses = []

    for epoch in range(pretrain_epochs):
        epoch_loss = 0
        train_range = tqdm(train_generator, total=len(train_generator))

        for i, (x, y,back_true) in enumerate(train_generator):
            if i >= pretrain_batches:
                break

            segm_mask=batched_segmentation(x).long()
            x,segm_mask = x.to(device), segm_mask.to(device) # Move to device the image and the segmentation mask
            pred = unet(x) # Forward pass with Unet
            loss = criterion_unet(pred, segm_mask.float()) # Compute loss BCElogits
            loss.backward()
            unet_optimizer.step()
            unet_optimizer.zero_grad()

            epoch_loss += loss.item()
            current_loss=epoch_loss/(i+1)
            train_range.set_postfix({'epoch': epoch+1,'loss': current_loss})  # Update the progress bar with the current loss
            train_range.update()
            #train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            #train_range.refresh()
   
        pretrain_losses.append(epoch_loss / pretrain_batches)
     # Delete tensors and free GPU memory
    del x, segm_mask, pred, loss
    torch.cuda.empty_cache()


    if fine_tuning:
        # Train unet and classifier on the remaining data
        for epoch in tqdm(range(1,train_epochs + 1)):
            train_loss = 0
            train_range = tqdm(train_generator)
            

            unet.train()
            classifier.train()
            for i, (x, y,back_true) in enumerate(train_range):
                if i < pretrain_batches:
                    continue
                
                x,back_true, y = x.to(device),back_true.to(device), y.to(device)
                pred = unet(x)
                #print("shape de pred:",pred.shape)
                #print("shape de y:",y.shape)
                #print("valeurs de y:",y)
            
                # Compute the loss and backpropagate through both networks
                class_pred = classifier(pred)
                loss = criterion_classif(class_pred, y) 
                loss.backward()
                classifier_optimizer.step()
                classifier_optimizer.zero_grad()

                # Now compute the loss again, this time only backpropagate through the U-Net
                class_pred = classifier(pred.detach())
                loss = criterion_classif(class_pred, y)
                loss.backward()
                unet_optimizer.step()
                unet_optimizer.zero_grad()

                train_loss += loss.item()

                current_loss = train_loss / (i + 1 - pretrain_batches)
                train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
                train_range.refresh()

            train_losses.append(train_loss / (n_train - pretrain_batches))

            # Validation
            # val_loss = 0
            # val_range = tqdm(val_generator)

            # unet.eval()
            # classifier.eval()
            # with torch.no_grad():
            #     for i, (x, y,back_true) in enumerate(val_range):
            #         x, y = x.to(device), y.to(device)
            #         pred = unet(x)
            #         class_pred = classifier(pred)
            #         loss = criterion(class_pred, y)
            #         val_loss += loss.item()

            #         current_loss = val_loss / (i + 1)
            #         val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f}")
            #         val_range.refresh()

            # val_losses.append(val_loss / n_val)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, pretrain_epochs + 1), pretrain_losses, label='Pretraining Loss')
    if fine_tuning:
        plt.plot(range(1, train_epochs + 1), train_losses, label='Training Loss')
    #plt.plot(range(1, config.learning.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #test on an image
    

    def test_model(unet, test_dataloader, device='cuda'):
        unet.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # No need to track gradients
            x,y,z = next(iter(test_dataloader))  # We take only the first batch
            x = x.to(device)  # Move the images to the device
            pred = unet(x)  # Apply the model
            #pred = torch.sigmoid(pred)  # Apply sigmoid to get pixel probabilities
            pred = (pred > 0.5).float()  # Binarize predictions to 0 and 1

            # Move the images and predictions to cpu
            x = x.cpu().numpy()
            pred = pred.cpu().numpy()

            # Plot the original image and the segmentation
            for i in range(min(10, x.shape[0])):  # We take only the first 10 images
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(np.transpose(x[i], (1, 2, 0)))  # Original image
                ax[1].imshow(np.transpose(pred[i], (1, 2, 0)))  # Segmentation
                plt.show()

    test_model(unet,test_dataloader)
     




if __name__=="__main__":
    unet=UNet()
    classifier=Classifier(num_classes=19)
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    train(config=config, unet=unet, classifier=classifier, k=0.8,pretrain_epochs=5,train_epochs=1,fine_tuning=False)  # Train the model

