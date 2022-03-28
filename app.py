import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

from flask import Flask, render_template, redirect, request


app = Flask(__name__)



model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))


def image_processing(image):

    #defining the size of the image
    img_shape = (321, 321)

    #classifier for the tensorflow module.
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(model_url, input_shape= img_shape + (3,), output_key="predictions:logits")])
    
    #opening the image and resizing it 
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img = np.array(img) / 255.0
    img = img[np.newaxis]
    
    #getting the predicted result.
    result = classifier.predict(img)
    return labels[np.argmax(result)]

def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)

    #returns the address of the place, latitude and the longitude.
    return [location.address,location.latitude, location.longitude]

def monu(image):
    monu_name  =  image_processing(image)
    monu_locate = get_map(monu_name)
    return [monu_name, monu_locate]

@app.route('/')

def index():
    lst = monu("download.jpg")
    return render_template("index.html", monu_list = lst)

if __name__ == '__main__':
    app.run(debug = True)