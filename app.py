from flask import Flask, render_template, json, jsonify, request, redirect, send_file
import os
import sys
import numpy as np
import cv2
import io
from base64 import b64decode, b64encode
from PIL import Image, ImageEnhance 
from io import BytesIO
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import keras
import matplotlib.pyplot as plt
from keras import backend as K

import tensorflow as tf
app = Flask(__name__,static_url_path='', static_folder='static', template_folder='templates')

global model

def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

umodel = tf.keras.models.load_model("models/unet_res", custom_objects={"mean_iou": mean_iou})

@app.route("/")
def index():
    return render_template('app.html')

@app.route("/analyze", methods=['GET','POST'])
def analyze():
    file = request.form['img']
    header, encoded = file.split(",", 1)
    data = b64decode(encoded)
    #i_img = cv2.imread("static/img/ckskdoojg4rmc0yafccv0bvwk.jpg")
    i_img = np.asarray(Image.open(io.BytesIO(data)))
    img = np.zeros([256, 256, 3], dtype=np.uint8)
    img_input = resize(i_img, (256, 256, 3), mode='constant', preserve_range=True)
    img = img_input
    # Convert to Tensor of type float32 for example
    image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # Add dimension to match with input mode 
    image_tensor = tf.expand_dims(image_tensor, 0)
    
    pred = umodel.predict(image_tensor, verbose=1)
    pred = np.argmax(pred, axis=3)[0,:,:]
    print(np.sum(pred == 1))
    pred = cv2.normalize(pred, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    pred = pred.astype(np.uint8)
    # pred = cv2.cvtColor(new_pred, cv2.COLOR_GRAY2RGB)
    print(np.unique(pred))
   
    pil_img = Image.fromarray(pred)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = b64encode(buff.getvalue()).decode("utf-8")

    count = 0.0
    area = 0.0

    count = np.sum(pred != 0)  #classification
    print(count)
    if round((count / pred.size)*100, 2) >= 8:
        viability = 'Very Viable'
    elif round((count / pred.size)*100, 2) >= 6:
        viability = 'Viable'
    elif round((count / pred.size)*100, 2) >= 4:
        viability = 'Less Viable'
    elif round((count / pred.size)*100, 2) >= 2:
        viability = 'Unviable'
    else:
        viability = 'Very Unviable'

    return jsonify({ 'o_image' : file, 'image': 'data:image/jpeg;base64,' + str(new_image_string), 'ppixel': str(round((count / pred.size)*100, 2)), 'viability': viability })


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=7000, debug=True)