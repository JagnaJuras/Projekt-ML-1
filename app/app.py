from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


model = load_model('../output/FishModelClassifier_V6.h5', compile=False)
class_name = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
              'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp', 'Green Spotted Puffer',
              'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish',
              'Long-Snouted Pipefish',
              'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
              'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']


def predict(path):
    img = load_img(path, target_size=(224, 224, 3))  # convert image size and declare the kernel
    img = img_to_array(img)  # convert the image to an image array
    img = img / 255  # rgb is 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    x = list(np.argsort(answer[0])[::-1][:5])

    for i in x:
        print("{className}: {predVal:.2f}%".format(className=class_name[i], predVal=float(answer[0][i]) * 100))

    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = class_name[y]

    return res



app = Flask(__name__, static_url_path='/static')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_species = predict(file_path)
        return render_template('index.html', prediction="Predicted Species: " + predicted_species, filename=filename)
    else:
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
