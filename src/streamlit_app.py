import streamlit as st
import os
import argparse
from PIL import Image
from torchvision import transforms
from easydict import EasyDict
import torch
import sys
from torch import Tensor
import cv2

#add the one level up directory to the sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import load_config, find_config
#from src.train import train_adversarial, test
from config.config import test_logger
from src.dataloader.dataloader import LABELS
from src.explainable.create_mask import mask_red_pixel_hsv
from metrics import Metrics
from src.model import resnet
from utils import utils
import numpy as np
from src.infer import infer
from PIL import ImageEnhance

st.title("Image Classification App")

st.sidebar.title("Image Transformations")

# Ajoutez ces lignes après les autres contrôles de la barre latérale
hue_min = st.sidebar.slider('Hue Min', min_value=0, max_value=180, value=0)
hue_max = st.sidebar.slider('Hue Max', min_value=0, max_value=180, value=180)
plot_mask = st.sidebar.checkbox('Plot Mask', value=False)


# Path to the training data
train_data_path = 'data/data_retouche/train_512'

#make a list of all the paths of the folders in data/data_labo the path should have the shape data/data_labo/train_512, data/data_labo/val_512, data/data_labo/test_512
list_paths = [os.path.join('data/data_labo', folder) for folder in os.listdir('data/data_labo')]

#make a choice box to select the path
train_data_path = st.selectbox("Select the training data path", list_paths)

# Get the list of directories in the training data path
#class_names = [name for name in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, name))]
class_names = LABELS
#print the class names
print(class_names)


# Recursive function to find .yaml files in a directory
def find_yaml_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml'):
                yield os.path.join(root, file)

# Menu for config selection
config_options = list(find_yaml_files('logs'))
config_file = st.selectbox("Select Config", config_options)


# Image upload
image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
print("IMAGE FILE: ", image_file)
if image_file is not None:
    image = Image.open(image_file)
    # Convertissez l'image PIL en tableau numpy pour l'utiliser avec OpenCV
    image_np = np.array(image)
    # Add sliders for image transformations
    rotation = st.sidebar.slider('Rotation', min_value=0, max_value=180, value=0)
    hflip = st.sidebar.checkbox('Horizontal Flip', value=False)
    vflip = st.sidebar.checkbox('Vertical Flip', value=False)
    contrast = st.sidebar.slider('Contrast', min_value=-3.0, max_value=3.0, value=1.0)
    brightness = st.sidebar.slider('Brightness', min_value=-3.0, max_value=3.0, value=1.0)


    #apply the transformations
    #Apply transformations to the image
    image = image.rotate(rotation)
    if hflip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(contrast)
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness)

    # Appliquez le masque si le bouton "plot mask" est coché
    if plot_mask:
        mask = mask_red_pixel_hsv(image_np, hue_min, hue_max)
    
        # Créez une image de superposition verte avec le masque
        overlay = np.zeros_like(image_np)
        overlay[mask > 0] = (0, 255, 0)  # Vert en RGB
        
        # Superposez le masque à l'image originale
        alpha = 0.5  # Définissez l'alpha pour la transparence
        image_np = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)
        
        # Convertissez l'image numpy en image PIL pour l'afficher avec Streamlit
        image = Image.fromarray(image_np)

    st.image(image, caption='Uploaded Image.', use_column_width=True)

# Inference button
if st.button("Inference"):
    if image_file is not None:
        # Load config
        config = load_config(config_file)

        # Get device
        device = utils.get_device(device_config=config.learning.device)

        print("config_file: ", config_file)
        config_in_log_dir = os.path.dirname(config_file)
        print("config_in_log_dir: ", config_in_log_dir)
        model = resnet.get_resnet(config)
        weight = utils.load_weights(config_in_log_dir, device=device)
        model.load_dict_learnable_parameters(state_dict=weight, strict=True)
        model = model.to(device)
        del weight


        # Prepare image
        transform = transforms.Compose([transforms.ToTensor()])
        #apply the transform to the image
        image = transform(image).unsqueeze(0).to(device)

        print("image.shape: ", image.shape)
        print("image", image)

        # Inference
        model.eval()
        with torch.no_grad():
            y_pred = model.forward(image)

        #st.balloons()

        # Display result
        res=class_names[y_pred.argmax(dim=-1).item()]
        st.success(f"**This image is classified as:** {res}")

