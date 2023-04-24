import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Define the window and widgets
window = tk.Tk()
window.title("Stroke Risk Predictor")
window.geometry("500x500")

gender_label = tk.Label(window, text="Gender (0 for Female, 1 for Male): ")
gender_label.pack()

gender_entry = tk.Entry(window)
gender_entry.pack()

age_label = tk.Label(window, text="Age (in years): ")
age_label.pack()

age_entry = tk.Entry(window)
age_entry.pack()

hypertension_label = tk.Label(window, text="Hypertension (0 for No, 1 for Yes): ")
hypertension_label.pack()

hypertension_entry = tk.Entry(window)
hypertension_entry.pack()

heart_disease_label = tk.Label(window, text="Heart Disease (0 for No, 1 for Yes): ")
heart_disease_label.pack()

heart_disease_entry = tk.Entry(window)
heart_disease_entry.pack()

ever_married_label = tk.Label(window, text="Ever Married (0 for No, 1 for Yes): ")
ever_married_label.pack()

ever_married_entry = tk.Entry(window)
ever_married_entry.pack()

work_type_label = tk.Label(window, text="Work Type (0 for Private, 1 for Self-employed, 2 for Govt_job, 3 for children, 4 for Never_worked): ")
work_type_label.pack()

work_type_entry = tk.Entry(window)
work_type_entry.pack()

residence_type_label = tk.Label(window, text="Residence Type (0 for Rural, 1 for Urban): ")
residence_type_label.pack()

residence_type_entry = tk.Entry(window)
residence_type_entry.pack()

avg_glucose_level_label = tk.Label(window, text="Average Glucose Level (in mg/dL): ")
avg_glucose_level_label.pack()

avg_glucose_level_entry = tk.Entry(window)
avg_glucose_level_entry.pack()

bmi_label = tk.Label(window, text="BMI: ")
bmi_label.pack()

bmi_entry = tk.Entry(window)
bmi_entry.pack()

smoking_status_label = tk.Label(window, text="Smoking Status (0 for never smoked, 1 for formerly smoked, 2 for smokes): ")
smoking_status_label.pack()

smoking_status_entry = tk.Entry(window)
smoking_status_entry.pack()

result_label = tk.Label(window, text="")
result_label.pack()

# Define the predict function
def predict():
    gender = int(gender_entry.get())
    age = int(age_entry.get())
    hypertension = int(hypertension_entry.get())
    heart_disease = int(heart_disease_entry.get())
    ever_married = int(ever_married_entry.get())
    work_type = int(work_type_entry.get())
    residence_type = int(residence_type_entry.get())
    avg_glucose_level = float(avg_glucose_level_entry.get())
    bmi = float(bmi_entry.get())
    smoking_status = int(smoking_status_entry.get())

    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })
    print(input_data)
    # Scale the input data using the same scaler used to train the model
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)
    print(input_data)
    if age>35 and avg_glucose_level>200:
        x=0.2
    else:
        x=0

    # Make the prediction using the pre-trained model
    prediction = model.predict(input_data)
    print(prediction)

    # Convert the prediction to a binary value (0 or 1)
    if prediction+x > 0.2:
        result = "at risk of stroke"
    else:
        result = "not at risk of stroke"

    # Update the result label
    result_label.config(text=result)

predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.pack()

window.mainloop()
