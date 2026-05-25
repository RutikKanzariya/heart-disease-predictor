import streamlit as st
# !pip install streamlit

import pandas as pd
import joblib
model = joblib.load("LR_heart_dieses.pkl")
scaler = joblib.load("scaler.pkl")
expected_col = joblib.load("X_Columns.pkl")
  

st.title("Heart Stroke prediction by RUTIK")
st.markdown("Provide the Follwing Details")
age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",[ "Male","Female"])
pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",80,200,100)
cholesterol = st.number_input("Cholesterol (mg/DL)",100,600,200)
fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/DL" ,[0,1])
resting_ecg = st.selectbox("Resting ECG",[ "Normal","ST","LVH"])
max_hr = st.slider("Max Heart Rate",60,220,150)
excercise_angina = st.selectbox("Excercise-Inducec Angina",[ "Y","N"])
oldpeak = st.slider("OldPeak (ST Depression)",0.0,6.0,1.0)  
st_slop = st.selectbox("ST Slope",[ "Up","Flat","Down"])
if st.button("Predict"):
    raw_input = {
        'Age' : age,
        'RestingBP' : resting_bp,
        'Cholesterol' :cholesterol,
        'FastingBS' : fastingbs,
        "MaxHR" : max_hr,
        'OldPeak' : oldpeak,
        'Sex_' + sex:1,
        'ChestPainType_' + pain:1,
        'RestingECG_' + resting_ecg :1,
        'ExerciseAngina_'+excercise_angina : 1,
        'ST_Slope_' + st_slop :1    
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_col:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_col]
    scaled_inp = scaler.transform(input_df)
    pred =  model.predict(scaled_inp)[0]
    
    if pred == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low risk of Heart Disease")
