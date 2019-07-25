from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from sklearn.externals import joblib
import numpy as np
import requests
import json

app = Flask(__name__)

from flask_sqlalchemy import SQLAlchemy
app.config.from_object("config.DevelopmentConfig")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

@app.route("/")
def index():

   response = make_response(render_template("index.html"))
   return response

@app.route("/predict", methods=['POST'])
def predict():
   if request.method=='POST':

       regressor = joblib.load("./linear_regression_model.pkl")
       data = dict(request.form.items())
       years_of_experience = np.array(float(data["YearsExperience"])).reshape(1,-1)
       prediction = regressor.predict(years_of_experience)
       response = make_response(render_template("predicted.html",prediction = float(prediction)))

       return response


if __name__ == '__main__':
   app.run(debug=True)