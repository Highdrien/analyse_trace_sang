import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from os.path import dirname as up

import torch
from torch import nn
from torch.optim import Adam

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.explainable.create_mask import batched_segmentation
from src.dataloader.dataloader import create_dataloader
from src.model.segmentation_model import UNet, Classifier
from utils import utils


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        return -(target * torch.log(input)).sum(dim=1).mean()


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        
        intersection = (output * target).sum()
        union = output.sum() + target.sum()
        dice = 2.0 * intersection / (union + self.eps)
        return 1.0 - dice


def train(config: EasyDict,
          unet: UNet,
          classifier: Classifier,
          alpha: float = 0.5
          ) -> None:
    
    if config.model.name != 'unet':
        raise ValueError(f"Expected model.name=unet but found {config.model.name}.")

    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    test_generator = create_dataloader(config=config, mode='test')
    n_train = len(train_generator)
    print(f"Found {n_train} training batches")

    # Number of classes
    num_classes = 4

    # Loss
    criterion_unet = DiceLoss()
    criterion_classif = torch.nn.CrossEntropyLoss()
    criterion_classif_2 = CustomCrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss(reduction='batchmean')

    # Optimizer
    unet_optimizer = Adam(unet.parameters(), lr=0.001)
    classifier_optimizer = Adam(classifier.parameters(), lr=0.001)

    # Get and put on device
    device = utils.get_device(device_config=config.learning.device)
    unet.to(device) 
    classifier.to(device)

    # Lists to store losses
    train_losses = []


    #debugging
    torch.autograd.set_detect_anomaly(True)

    # Create a batch of uniform distributions
    batch_size = 8
    uniform_distribution = torch.ones(batch_size, num_classes).to(device) / num_classes  # create a uniform distribution
        

    # Train unet and classifier on the data
    for epoch in tqdm(range(1, config.learning.epochs + 1)):
        train_loss = 0
        train_range = tqdm(train_generator)

        unet.train()
        classifier.train()

        # Avant la boucle d'entraînement, on met les gradients à zéro
        unet_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        for i, (x, y, back_true) in enumerate(train_range):
            segm_mask = batched_segmentation(x).long()
            x, back_true, y = x.to(device), back_true.to(device), y.to(device)

            # Prédiction du masque à partir de x
            pred = unet(x)
            segm_mask = segm_mask.to(device).float()

            # Calcul de la loss entre le masque prédit et le vrai masque (dice loss)
            loss_unet = criterion_unet(pred, segm_mask)
            loss_unet.backward(retain_graph=True)  # Rétropropagation de la loss
            unet_optimizer.step()  # Mise à jour des poids du unet
            unet_optimizer.zero_grad()  # Remise à zéro des gradients du unet

            # Multiplication de x par le masque prédit
            roi = x * pred  # .detach() pour ne pas suivre les gradients
            # Par celle-ci
            

            # Prédiction de la classe à partir du produit
            class_pred = classifier(roi)

            # Calcul de la loss du classifieur
            loss_classif = criterion_classif(class_pred, back_true)
            loss_classif.backward(retain_graph=True)  # Rétropropagation de la loss
            classifier_optimizer.step()  # Mise à jour des poids du classifieur
            classifier_optimizer.zero_grad()  # Remise à zéro des gradients du classifieur

            # Calcul de la loss finale
            loss = loss_unet.detach() + alpha * criterion_classif_2(class_pred, uniform_distribution)
            loss.backward()  # Rétropropagation de la loss
            unet_optimizer.step()  # Mise à jour des poids du unet
            classifier_optimizer.step()  # Mise à jour des poids du classifieur

            train_loss += loss.item()
            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()

        train_losses.append(train_loss / n_train)


    def test_model(unet, test_dataloader, device='cuda'):
        unet.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # No need to track gradients
            x, y, z = next(iter(test_dataloader))  # We take only the first batch
            x = x.to(device)  # Move the images to the device
            pred = unet(x)  # Apply the model
            pred = torch.sigmoid(pred)  # Apply sigmoid to get pixel probabilities
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

    test_model(unet,test_generator)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, config.learning.epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__=="__main__":



    #test the custom loss
    #y_pred=torch.tensor([[[[0.1,0.2],[0.3,0.4]]]])
    #y_true=torch.tensor([[[[0,1],[1,0]]]])
    #print("cutsom crossentropy loss:",CustomCrossEntropyLoss()(y_pred,y_true))



    unet = UNet()
    classifier = Classifier(num_classes=4)
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    config.model.name = 'unet'
    train(config=config, unet=unet, classifier=classifier, alpha=0.3)  # Train the model