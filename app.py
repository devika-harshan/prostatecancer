import joblib
from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)


model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template("cancer.html")



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        rad = float(request.form['radius'])
        tex = float(request.form['texture'])
        par = float(request.form['perimeter'])
        area = float(request.form['area'])
        smooth = float(request.form['smoothness'])
        compact = float(request.form['compactness'])
        symme= float(request.form['symmetry'])
        frac = float(request.form['fractal_dimension'])

        mypred = np.array([[rad, tex, par, area, smooth, compact,symme, frac]])
        my_prediction = model.predict(mypred)

        return render_template('cancerresult.html', prediction=my_prediction)

  

if __name__ == "__main__":
    app.run(debug=True)
