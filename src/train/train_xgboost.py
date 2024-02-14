import os
import sys
import shap
import yaml
import numpy as np
from PIL import Image
import xgboost as xgb
from easydict import EasyDict
import matplotlib.pyplot as plt
from os.path import dirname as up

sys.path.append(up(up(up(os.path.abspath(__file__)))))


from src.dataloader.dataloader import create_dataloader
from src.explainable.create_mask import segment_image_file
from src.explainable.find_criteres import Critieres

def extract_features_and_labels(generator):
    features = []
    labels = []
    criteres = Critieres()

    for i, (batch_x, batch_y, _) in enumerate(generator):
        print("Treating batch:", i, "out of", len(generator))
        for image, label in zip(batch_x, batch_y):
            image=image.permute(1, 2, 0).numpy()
            image = image * 255
            # Segment the image
            mask = segment_image_file(image)

            #inverser les valeurs de la mask
            mask = 1 - mask

            mask = mask[:,:,0]
            feature = criteres.get_critieres(mask)
            features.append(feature)
            labels.append(label)

    return np.array(features), np.array(labels)

def train_xgboost(dataloader):
    model = xgb.XGBClassifier(tree_method='gpu_hist')
    features, labels = extract_features_and_labels(dataloader)
    model.fit(features, labels)
    return model

def test_xgboost(model, dataloader):
    test_features, test_labels = extract_features_and_labels(dataloader)
    accuracy = model.score(test_features, test_labels)
    print("Test Accuracy: ", accuracy)
    return accuracy


def plot_shap_values(model, dataloader):
    # Load all the images
    images = []
    for i, (batch_x, batch_y, _) in enumerate(dataloader):
        for image, label in zip(batch_x, batch_y):
            images.append(image.permute(1, 2, 0).numpy()*255) #attention au *255 !!!

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    print("EXPLAINER created")
    criteres = Critieres()

    for k in images:
        print("Treating image:", k)

        # Segment the image
        mask = segment_image_file(k)

        #Delete the last dimension of the mask
        mask = mask[:,:,0]

        #inverser les valeurs de la mask
        mask = 1 - mask
        feature = criteres.get_critieres(mask)

        # Append the features and label to their respective lists
        features = np.array(feature)

        # Compute XGBoost prediction
        prediction = model.predict(np.reshape(features, (1, -1)))
        print("Max value of prediction:",prediction.max())
        print("XGBoost prediction:", prediction)
        #print("The image belong to model:", ["Passive", "Glissée", "impact"][prediction[0]])
        #predicted_class = ["Passive", "Glissée", "impact"][prediction[0]]
        # Get the SHAP values for the first prediction
        shap_values = explainer.shap_values(np.reshape(features, (1, -1)))
        print("SHAP values:", shap_values)

        # Plot the SHAP values with a bar plot
        # Create a new figure with 2 subplots
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        # Add a title to the figure with the predicted class
        #fig.suptitle(f'The prediction of the model is: {predicted_class}', fontsize=16)

        # Plot the image on the first subplot
        axs[0].imshow(k/255)
        axs[0].set_title('Processed Image')

        # Plot the SHAP values with a bar plot on the second subplot
        axs[1].bar(["Ovality", "Satellites", "Irregularity", "Satellite Ratio", "Homogeneity", "Striation", "Distribution"], shap_values[0][0])
        axs[1].set_title('SHAP Values')

        # Plot the segmented image on the third subplot
        axs[2].imshow(mask, cmap='gray')
        axs[2].set_title('Segmented Image')

        # Display the plots
        plt.show()

def inference_xgboost(model, image):
    """
    input: image, a numpy array
    output: the prediction of the model
    """
    # Segment the image
    mask = segment_image_file(image)
    criteres = Critieres()

    #inverser les valeurs de la mask
    mask = 1 - mask

    #multiplier par 255
    mask = mask * 255
    feature = criteres.get_critieres(mask)
    # Append the features and label to their respective lists
    features = np.array(feature)

    # Compute XGBoost prediction
    prediction = model.predict(np.reshape(features, (1, -1)))
                    
    #compute the SHAP values
    # Create a SHAP explainers
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(np.reshape(features, (1, -1)))


    return prediction, shap_values


def save_xgboost_model(model, path):
    """
    Save the XGBoost model to the specified path and create the directory if it does not exist
    """
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the model
    model.save_model(path)

def load_xgboost_model(path):
    """
    Load the XGBoost model from the specified path and return it
    """
    # Create a new XGBoost model
    model = xgb.XGBClassifier()

    # Load the model
    model.load_model(path)

    return model

if __name__ == "__main__":

    #config
    config=EasyDict(yaml.safe_load(open('config/config.yaml')))

    #create a dataloader
    train_generator = create_dataloader(config=config, mode='train')

    #create a xgboost model
    model = train_xgboost(train_generator)

    #create a test generator
    test_generator = create_dataloader(config=config, mode='test')

    #evaluate the model on the test set
    test_xgboost(model, test_generator)

    #plot the SHAP values
    plot_shap_values(model, test_generator)