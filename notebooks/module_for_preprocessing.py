# Dans ce module, on dééfinit les fonctions qui nous serons utile pour le preprocessing.py


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as app
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image

import os
import yaml
import numpy as np
from PIL import Image




with open('../config.yml', 'r') as file:
    config = yaml.safe_load(file)


image_size = config['donnees']['image']['size']
normalize = config['donnees']['image']['normalize']
standardize = config['donnees']['image']['standardize']
grayscale = config['donnees']['image']['grayscale']

def resize_and_rename_images_of_folder(folder,subfolder,size):
       data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), "data", folder, subfolder)
       resized = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), "data", folder, f"resized_{subfolder}_{size}")
       os.makedirs(resized, exist_ok=True) # au cas ou 


       for count,filename in enumerate(os.listdir(data_path)):
        image_path = os.path.join(data_path, filename)
        image = Image.open(image_path)
        image_resized = image.resize((size,size))
        output_path = os.path.join(resized, f'{subfolder}_{count}.jpg')
        image_resized.save(output_path)


       

def load_and_preprocess_image(image,grayscale,normalize,standardize):
    # Charger l'image
    img = tf.io.read_file(image)
    img = tf.image.decode_image(img, channels=1 if grayscale else 3)  # Convertir en niveaux de gris si spécifié

    if normalize:
        img = img / 255.0  # Normaliser entre 0 et 1

    if standardize:
        img = (img - tf.reduce_mean(img)) / tf.math.reduce_std(img)  # Standardisation

    return img

def load_images_with_preprocessing(directory, size):

    images = []
    labels = []

    subfolders = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    class_names = [f"resized_{subfolder}_{size}" for subfolder in subfolders]

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            img = Image.open(img_path).convert('L')  # Convertir en niveaux de gris 
            img_array = np.array(img)  
            
            img_array = np.expand_dims(img_array, axis=-1)

            images.append(img_array)
            labels.append(label)


    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)