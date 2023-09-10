
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS

import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)

# load model for prediction
modelvgg = load_model("modelVGG_sec_1.h5", compile=False)
modelxception = load_model("modelXCEPTION_sec_1.h5", compile=False)


UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
        # img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # convert image to PNG format
    # img = Image.open(img_url)
    # now = datetime.now()
    # predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    # img.save(predict_image_path)  
    # img.close()

    # prepare image for prediction
    img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(224, 224, 3))
    x = image.img_to_array(img)
    x = x/127.5-1 
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # predict
    prediction_array_vgg = modelvgg.predict(images)
    prediction_array_xception = modelxception.predict(images)

    # prepare api response
    class_names = ['alif', 'ba', 'ta', 'tsa', 'jim', 'ha', 'kha', 'dal', 'dzal', 'ra', 'zai', 'sin', 'syin', 'shod', 'dhod', 'tho', 'dzo', 'ain', 'ghain', 'fa', 'qof', 'kaf', 'lam', 'mim', 'nun', 'waw', 'haa', 'lamalif', 'hamzah', 'ya']
	
    # Generate voice output
    voice_text = f"VGG Prediction is: {class_names[np.argmax(prediction_array_vgg)]}. "
    voice_text += f"Xception Prediction is: {class_names[np.argmax(prediction_array_xception)]} ."
    tts = gTTS(text=voice_text, lang='ar')
    tts.save('static/voice/output.mp3')

    # Mendapatkan format file gambar dari nama file
    image_format = filename.split('.')[-1]


    return render_template("output.html", img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename),
                        predictionvgg = class_names[np.argmax(prediction_array_vgg)],
                        predictionxception = class_names[np.argmax(prediction_array_xception)],
                        image_format=image_format)

if __name__ =='__main__':
	#app.debug = True
	app.run(port = 8000, debug=True)