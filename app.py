from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the request
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    gender = request.form['gender']
    ever_married = request.form['ever_married']
    work_type = request.form['work_type']
    residence_type = request.form['residence_type']
    smoking_status = request.form['smoking_status']

    # Create a new dataframe with the form data
    data = {'age': [age], 'hypertension': [hypertension], 'heart_disease': [heart_disease], 'avg_glucose_level': [avg_glucose_level], 'bmi': [bmi], 'gender_' + gender: [1], 'ever_married_' + ever_married: [1], 'work_type_' + work_type: [1], 'residence_type_' + residence_type: [1], 'smoking_status_' + smoking_status: [1]}
    df = pd.DataFrame(data)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    if age>35 and avg_glucose_level>200:
        x=0.2
    else:
        x=0

    # Make a prediction using the trained model
    prediction = model.predict(X)[0][0]
    if prediction+x > 0.2:
        result = "High"
    else:
        result = "Low"

    # Render the prediction result on a new page
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
