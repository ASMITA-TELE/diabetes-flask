Diabetes Prediction Web App

A web-based application to predict whether a person is diabetic or non-diabetic using a machine learning model trained on the PIMA Diabetes Dataset. Built with Python, Flask, and scikit-learn.

Features

User-friendly web interface to input patient data.

Uses Support Vector Machine (SVM) classifier for prediction.

Standardizes input features before prediction.

Displays prediction results on a new webpage.

Easily extensible to other ML models or datasets.

Dataset

Dataset: PIMA Diabetes Dataset

Source: Kaggle Diabetes Dataset

Features:

Feature	Description
Pregnancies	Number of times pregnant
Glucose	Plasma glucose concentration (mg/dL)
BloodPressure	Diastolic blood pressure (mm Hg)
SkinThickness	Triceps skinfold thickness (mm)
Insulin	2-Hour serum insulin (mu U/ml)
BMI	Body mass index (weight in kg / height² in m²)
DiabetesPedigreeFunction	Diabetes pedigree function
Age	Age in years

Target: Outcome (0 = Non-Diabetic, 1 = Diabetic)
