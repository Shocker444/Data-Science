import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

os_path = Path(__file__).parents[0]



st.set_page_config(layout='wide')


st.title('Diabetes prediction app')

gender = st.number_input(label='Input your gender in numbers(0-female, 1-male)')

age = st.number_input(label='Input your Age')

hypertension = st.number_input(label='Any previous case of hypertension?(0-No, 1-Yes)')

hrt_disease = st.number_input(label='Previous case of heart disease(0-No, 1-Yes)')

smoking_hist = st.number_input(label='Affiliated with smoking?(0-No Info, 1-Current, 2-Still smoking, 3-Not anymore, 4-Never, 5-Not currently)')

bmi = st.number_input(label='Input your Basic Mass Index(BMI)')

HbA1c_level = st.number_input(label='Input your HbA1c level')

blood_glucose_level = st.number_input(label='Input your blood glucose level')

instance = np.array([[gender, age, hypertension, hrt_disease,
                      smoking_hist, bmi, HbA1c_level, blood_glucose_level]])


def generate_results(inst):
    classifier = joblib.load(f"{os_path}/classifier.sav")
    pred = classifier.predict(inst)
    if pred == 0:
        st.write('No Diabetes tendencies present')
    else:
        st.write('Diabetes present. Get medical attention immediately')

if st.button('Generate results', type='primary'):
    generate_results(instance)

