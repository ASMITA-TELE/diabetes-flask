from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# load model + scaler (files must be in project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # read form inputs (names must match the index.html inputs)
    try:
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age']),
        ]
    except Exception as e:
        return f"Invalid input: {e}", 400

    arr = np.array(features).reshape(1, -1)
    arr_std = scaler.transform(arr)
    pred = model.predict(arr_std)[0]
    result_text = "Diabetic" if pred == 1 else "Not Diabetic"
    return render_template('result.html', result=result_text)

if __name__ == '__main__':
    # You can run this file directly: python app.py
    app.run(debug=True)
