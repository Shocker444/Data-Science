import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


st.set_page_config(layout='wide')


st.title('Diabetes prediction app')

gender = st.number_input(label='Input your gender in numbers(0-female, 1-male)')
st.write(gender)

age = st.number_input(label='Input your Age')
st.write(age)

hypertension = st.number_input(label='Any previous case of hypertension?(0-No, 1-Yes)')
st.write(hypertension)

hrt_disease = st.number_input(label='Previous case of heart disease(0-No, 1-Yes)')
st.write(gender)

smoking_hist = st.number_input(label='Affiliated with smoking?(0-No Info, 1-Current, 2-Still smoking, 3-Not anymore, 4-Never, 5-Not currently)')
st.write(smoking_hist)

bmi = st.number_input(label='Input your Basic Mass Index(BMI)')
st.write(bmi)

HbA1c_level = st.number_input(label='Input your HbA1c level')
st.write(HbA1c_level)

blood_glucose_level = st.number_input(label='Input your blood glucose level')
st.write(blood_glucose_level)

instance = np.array([[gender, age, hypertension, hrt_disease,
                      smoking_hist, bmi, HbA1c_level, blood_glucose_level]])
instance = StandardScaler().fit_transform(instance)


def generate_results(inst):
    classifier = joblib.load('./classifier.sav')
    pred = classifier.predict(inst)
    if pred == 0:
        st.write('No Diabetes tendencies present')
    else:
        st.write('Diabetes present. Get medical attention immediately')

if st.button('Generate results', type='primary'):
    generate_results(instance)

