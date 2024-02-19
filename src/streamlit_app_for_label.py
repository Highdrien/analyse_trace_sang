import streamlit as st
import os
import numpy as np

from PIL import Image
import shutil
import sys

#add the one level up directory to the sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.explainable.create_mask import mask_red_pixel
from PIL import Image


import sys



#from src.train import train_adversarial, test


# Fonction pour obtenir tous les chemins d'images dans un dossier et ses sous-dossiers
def get_image_paths(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
                print(image_paths[-1])
    return image_paths

# Fonction pour afficher l'image et son masque
def display_image_and_mask(image_path):
    print(image_path)
    image = Image.open(image_path)
    image=np.array(image)
    mask = mask_red_pixel(image)  # Assurez-vous que cette fonction est correctement définie et importée
    mask=mask*255
    st.image([image, mask], width=300)

# Fonction pour enregistrer l'image et son masque
def save_image_and_mask(image_path, train_dir):
    data_dir = os.path.join(train_dir, 'data')
    label_dir = os.path.join(train_dir, 'label')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    shutil.copy(image_path, data_dir)

    image = Image.open(image_path)
    image_array = np.array(image)
    mask = mask_red_pixel(image_array)  # Pass the image array instead of the path

    # Ensure the mask is a 2D array with data type uint8
    mask = np.squeeze(mask).astype('uint8')

    #Multiply the mask by 255
    mask=mask*255

    mask_image = Image.fromarray(mask)  # Convert the mask back to an Image object

    # Save the image in JPEG format
    mask_image.save(os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.jpg'), 'JPEG')

folder_path = st.text_input('Enter the folder path')

if 'image_paths' not in st.session_state:  # Initialize image_paths in session state if not already present
    st.session_state.image_paths = []

if st.button('Load Images'):
    st.session_state.image_paths = get_image_paths(folder_path)
    print("len image paths", len(st.session_state.image_paths))
    st.session_state.image_index = 0  # Initialize the image index

if 'image_index' in st.session_state:  # If the image index is defined
    display_image_and_mask(st.session_state.image_paths[st.session_state.image_index])  # Display the current image

    if st.button('Keep'):
        print("CURRENT IMAGE PATH", st.session_state.image_paths[st.session_state.image_index])
        save_image_and_mask(st.session_state.image_paths[st.session_state.image_index], 'train')
        st.session_state.image_index += 1  # Move to the next image
    if  st.button('Not Keep'):
        st.session_state.image_index += 1  # Move to the next image