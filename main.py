import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from slice_finder import SliceFinder
from slice_finder import SliceFinder2
import time

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory
import os

app = Flask(__name__)

FILENAME = None
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'output/'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/checked', methods = ['POST'])  
def accepted():  
    if request.method == 'POST':
        global FILENAME  
        f = request.files['csv-file']
        if f.filename == "":
            return render_template('index.html')
        f.save(secure_filename(f.filename))
        df = pd.read_csv(f.filename)
        FILENAME = f.filename
        if 'Target' in df.columns and 'Prediction' in df.columns:
            return render_template('accepted.html', name = f.filename,
                default_k=5, default_t=0.4)
        else:
            return render_template('unsupported.html', name = f.filename)

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    # Appending app path to upload folder path within app root folder
    uploads = os.path.join(APP_ROOT, app.config['UPLOAD_FOLDER'])
    # Returning file from appended path
    return send_from_directory(directory=uploads, filename=filename)

@app.route('/about')
def about():
    return render_template('about_us.html')

@app.route('/run', methods=['GET','POST'])
def run():
    sliderk = 5
    slidert = 0.4
    if request.method == 'POST':
        sliderk = int(request.form['sliderk'])
        slidert = float(request.form['slidert'])

    t0 = time.time()

    INPUT_FILE = FILENAME
    NUM_OF_SLICES = sliderk
    THRESHOLD_EFFECT_SIZE = slidert
    
    input_data = pd.read_csv(INPUT_FILE, header=0, sep='\s*,\s*', engine='python')
    input_data = input_data.dropna()

    predicted = input_data['Prediction']
    input_data = input_data[input_data.columns.difference(['Prediction'])]

    encoders = {}
    for column in input_data.columns:
        if input_data.dtypes[column] == object:
            le = LabelEncoder()
            input_data[column] = le.fit_transform(input_data[column])
            encoders[column] = le

    features = input_data[input_data.columns.difference(['Target'])]
    ground = input_data['Target']

    sf = SliceFinder2(features, ground, predicted)
    recommendations = sf.find_slice_2(k=NUM_OF_SLICES, epsilon=THRESHOLD_EFFECT_SIZE,
         degree=2, max_workers=4)
    my_slices = []

    for s in recommendations:
        my_slice = [None, None, None, None]
        my_pairs = []
        for k, v in list(s.filters.items()):
            my_pair = [None, None]
            values = ''
            if k in encoders:
                le = encoders[k]
                for v_ in v:
                    values += '%s ' % (le.inverse_transform(v_)[0])
            else:
                for v_ in sorted(v, key=lambda x: x[0]):
                    if len(v_) > 1:
                        values += '%s ~ %s' % (v_[0], v_[1])
                    else:
                        values += '%s ' % (v_[0])
            my_pair[0] = k
            my_pair[1] = values
            my_pairs.append(my_pair)
        my_slice[0] = my_pairs
        my_slice[1] = s.effect_size
        my_slice[2] = s.metric
        my_slice[3] = s.size
        my_slices.append(my_slice)

    df = pd.DataFrame(my_slices, columns=['Feature Value Pairs', 'Effect Size', 'Metric', 'Size'])
    print(df)

    t1 = time.time() - t0
    print("Time elapsed: ", t1, "seconds")
    df.to_csv("output/output.csv")
    data_temp = df.to_dict(orient='records')
    return render_template('output.html', name=FILENAME, data_2=data_temp,
     default_k=sliderk, default_t=slidert)

if __name__ == '__main__':
  app.run(debug=True)