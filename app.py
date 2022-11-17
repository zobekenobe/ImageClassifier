import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
import os

from tensorflow.keras.applications import (
    vgg16, mobilenet_v2, resnet50,inception_v3,vgg19
)

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def choose_model(model):
    if model.lower() == 'vgg16':
        return vgg16.VGG16(weights = 'imagenet', include_top = True)
    elif model.lower() == 'mobilenetv2':
        return mobilenet_v2.MobileNetV2(weights = 'imagenet', include_top = True)
    elif model.lower() == 'resnet50':
        return resnet50.ResNet50(weights = 'imagenet', include_top = True)
    elif model.lower() == 'inceptionv3':
        return inception_v3.InceptionV3(weights = 'imagenet', include_top = True)
    elif model.lower() == 'vgg19':
        return tf.keras.applications.vgg19.VGG19(weights = 'imagenet', include_top = True)
    else:
        print('Model not found')
        return 0

def classification_method(selection):
    model = choose_model(selection)
    # load the image 
    image = load_img(os.path.join(uploaded_file.name), target_size = (224, 224))
    # convert the image to an array
    image = img_to_array(image)
    # expand the dimensions to include the batch
    image = np.expand_dims(image, axis = 0)
    # preprocess the image for the model
    image = tf.keras.applications.vgg16.preprocess_input(image)
    # feed to the model
    predictions = model.predict(image)
    # decode the predictions
    label = tf.keras.applications.vgg16.decode_predictions(predictions)
    return label


st.title("Deep Learning Image Classifier using 'ImageNet' dataset ")

selection = st.sidebar.selectbox(label = 'Choose your model', options = ['VGG16','VGG19' ,'ResNet50', 'MobileNetV2', 'InceptionV3'])
submit = st.sidebar.button('Classify')

uploaded_file = st.sidebar.file_uploader(label = 'Upload the image')
if uploaded_file == None:
    st.stop()
st.image(uploaded_file)



label = classification_method(selection)
if submit:
    st.sidebar.success(f'The image is classified as {label[0][0][1]} with a confidence of {label[0][0][-1]*100:0.3f}%')







