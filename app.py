import PIL
import folium
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

from geopy.geocoders import Nominatim
from werkzeug.utils import secure_filename
from flask import Flask, render_template, redirect, request

app = Flask(__name__)

model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))
global monu_details 
monu_details = ["-","-","-"]

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
    return [location.address,location.latitude, location.longitude]

def monu(image):
    monu_name  =  image_processing(image)
    monu_locate = get_map(monu_name)
    return [monu_name, monu_locate]

@app.route('/')
def index():
    return render_template("index.html",monu_details = monu_details)


@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        content = request.files['file']
        content.save(secure_filename(content.filename))
        try:
            monu_details = monu(content)
            start_coords = (monu_details[1][1], monu_details[1][2])
            folium_map = folium.Map(location=start_coords, zoom_start=14)
            folium_map.save('templates/map.html')
            return render_template("index.html", monu_details = monu_details)
        except:
            return render_template("error.html")
    return render_template("index.html", monu_details = monu_details)



@app.route('/map', methods = ['GET', 'POST'])
def map_init():
    return render_template("map.html")

if __name__ == '__main__':
    app.run(debug = True)

