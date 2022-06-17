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
        frac = float(request.form['fractional_dimension'])

        data = np.array([[rad, tex, par, area, smooth, compact, frac]])
        my_prediction = model.predict(data)

        return render_template('cancerresult.html', prediction=my_prediction)

    # input_features = [int(x) for x in request.form.values()]
    # features_value = [np.array(input_features)]
    # features_name = ['radius', 'texture', 'perimeter', 'area',
    #                  'smoothness', 'compactness', 'symmetry', 'fractal_dimension']

                     
    # df = pd.DataFrame(features_value, columns=features_name)
    # output = model.predict(df)
    # if output == 4:
    #     res_val = "a high risk of Prostrate Cancer"
    # else:
    #     res_val = "a low risk of Prostrate Cancer"

    # return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))





if __name__ == "__main__":
    app.run(debug=True)