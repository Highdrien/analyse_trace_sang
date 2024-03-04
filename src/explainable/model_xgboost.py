import os
import shap
import yaml
import numpy as np
import xgboost as xgb
from easydict import EasyDict
import matplotlib.pyplot as plt

#add the src directory to the path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from src.dataloader.dataloader import create_dataloader
from src.explainable.create_mask import segment_image_file
from src.explainable.find_criteres import calculate_ovality, count_satellites, \
                                          calculate_irregularity, calculate_satellite_ratio, \
                                            calculate_homogeneity, count_internal_striations, classify_distribution

def extract_features_and_labels(generator):
    features = []
    labels = []
    for i, (batch_x, batch_y, _) in enumerate(generator):
        print("Treating batch:", i, "out of", len(generator))
        for image, label in zip(batch_x, batch_y):

            #get the image in numpy array
            image=image.permute(1, 2, 0).numpy()

            #mulitply the image by 255
            image = image*255

            #print the max values of the image
            #print("MAX VALUES:",image.max())


            # Segment the image
            mask = segment_image_file(image)

            #inverser les valeurs de la mask
            mask = 1-mask

            #print("Shape of the mask:",mask.shape)
            #the shape is (128,128,1) so we need to reshape it to (128,128) just suppress the last dimension
            mask = mask[:,:,0]


            # Extract features
            ovality = calculate_ovality(mask)
            satellites = count_satellites(mask)
            irregularity = calculate_irregularity(mask)
            ratio = calculate_satellite_ratio(mask)
            homogeneity=calculate_homogeneity(mask)
            striation=count_internal_striations(mask)
            distrib=classify_distribution(mask)


            # Append the features and label to their respective lists
            features.append([ovality, satellites, irregularity, ratio, homogeneity, striation, distrib])
            labels.append(label)

    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def train_xgboost(dataloader):
    # Create a xgboost model
    model = xgb.XGBClassifier(tree_method='gpu_hist') #GPU acceleration !

    # Extract the features and labels from the images
    features, labels = extract_features_and_labels(dataloader)

    # Train the model
    model.fit(features, labels)

    return model

def test_xgboost(model, dataloader):
    # Extract the features and labels from the images
    test_features, test_labels = extract_features_and_labels(dataloader)

    # Calculate accuracy
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

    for k in images:
        print("Treating image:", k)

        # Segment the image
        mask = segment_image_file(k)

        #Delete the last dimension of the mask
        mask = mask[:,:,0]

        #inverser les valeurs de la mask
        mask = 1-mask

        # Extract features
        ovality = calculate_ovality(mask)
        satellites = count_satellites(mask)
        irregularity = calculate_irregularity(mask)
        ratio = calculate_satellite_ratio(mask)
        homogeneity = calculate_homogeneity(mask)
        striation = count_internal_striations(mask)
        distrib = classify_distribution(mask)


        # Append the features and label to their respective lists
        features = np.array([ovality, satellites, irregularity, ratio, homogeneity, striation, distrib])

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

    #inverser les valeurs de la mask
    mask = 1-mask

    #multiplier par 255
    mask = mask*255


    # Extract features
    ovality = calculate_ovality(mask)
    satellites = count_satellites(mask)
    irregularity = calculate_irregularity(mask)
    ratio=calculate_satellite_ratio(mask)
    distrib=classify_distribution(mask)
    striation=count_internal_striations(mask)
    homogeneity=calculate_homogeneity(mask)



    

    # Append the features and label to their respective lists
    features = np.array([ovality, satellites, irregularity, ratio, homogeneity, striation, distrib])

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