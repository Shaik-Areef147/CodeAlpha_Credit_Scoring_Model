from flask import Flask
from flask import render_template
from flask import request

import joblib
import numpy as np

app = Flask(__name__)

import joblib

model = joblib.load("risk_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    features = [
        float(x)
        for x in request.form.values()
    ]

    prediction = model.predict(
        [features]
    )

    result = "Good Credit"

    if prediction[0] == 1:
        result = "Bad Credit"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)