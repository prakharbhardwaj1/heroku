# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:38:55 2020

@author: Prakhar
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('webpage.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('webpage.html', prediction_text='Employee seems to be: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
