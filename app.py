# import streamlit as st
# # !pip install streamlit

# import pandas as pd
# import joblib
# model = joblib.load("LR_heart_dieses.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_col = joblib.load("X_Columns.pkl")
  

# st.title("Heart Stroke prediction by RUTIK")
# st.markdown("Provide the Follwing Details")
# age = st.slider("Age",18,100,40)
# sex = st.selectbox("SEX",[ "Male","Female"])
# pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",80,200,100)
# cholesterol = st.number_input("Cholesterol (mg/DL)",100,600,200)
# fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/DL" ,[0,1])
# resting_ecg = st.selectbox("Resting ECG",[ "Normal","ST","LVH"])
# max_hr = st.slider("Max Heart Rate",60,220,150)
# excercise_angina = st.selectbox("Excercise-Inducec Angina",[ "Y","N"])
# oldpeak = st.slider("OldPeak (ST Depression)",0.0,6.0,1.0)  
# st_slop = st.selectbox("ST Slope",[ "Up","Flat","Down"])
# if st.button("Predict"):
#     raw_input = {
#         'Age' : age,
#         'RestingBP' : resting_bp,
#         'Cholesterol' :cholesterol,
#         'FastingBS' : fastingbs,
#         "MaxHR" : max_hr,
#         'OldPeak' : oldpeak,
#         'Sex_' + sex:1,
#         'ChestPainType_' + pain:1,
#         'RestingECG_' + resting_ecg :1,
#         'ExerciseAngina_'+excercise_angina : 1,
#         'ST_Slope_' + st_slop :1    
#     }
#     input_df = pd.DataFrame([raw_input])
#     for col in expected_col:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     input_df = input_df[expected_col]
#     scaled_inp = scaler.transform(input_df)
#     pred =  model.predict(scaled_inp)[0]
    
#     if pred == 1:
#         st.error("High Risk of Heart Disease")
#     else:
#         st.success("Low risk of Heart Disease")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #0E1117;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            font-size: 18px;
            font-weight: bold;
        }

        .title {
            text-align: center;
            color: #FF4B4B;
            font-size: 42px;
            font-weight: bold;
        }

        .subtext {
            text-align: center;
            font-size: 18px;
            color: #CCCCCC;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------
model = joblib.load("LR_heart_dieses.pkl")
scaler = joblib.load("scaler.pkl")
expected_col = joblib.load("X_Columns.pkl")

# --------------------------------------------------
# TITLE SECTION
# --------------------------------------------------
st.markdown('<p class="title">❤️ Heart Disease Prediction App</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtext">AI-Powered Heart Disease Risk Prediction System by Rutik</p>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("📌 About")

    st.write(
        """
        This AI system predicts the probability of heart disease
        using Machine Learning.

        ### Features
        - Logistic Regression Model
        - Real-time Prediction
        - Risk Probability
        - Interactive Dashboard
        - Clean UI
        """
    )

    st.info("⚠️ This app is for educational purposes only.")

    st.write("---")

    st.write(f"🕒 {datetime.now().strftime('%d %B %Y')}")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.subheader("🩺 Patient Health Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    pain = st.selectbox(
        "Chest Pain Type",
        ['ATA', 'NAP', 'TA', 'ASY']
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

with col2:
    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
    )

    fastingbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

with col3:
    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "OldPeak (ST Depression)",
        0.0,
        6.0,
        1.0,
        0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
if st.button("🔍 Predict Heart Disease Risk"):

    # ----------------------------------------------
    # RAW INPUT
    # ----------------------------------------------
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingbs,
        'MaxHR': max_hr,
        'OldPeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # ----------------------------------------------
    # CREATE DATAFRAME
    # ----------------------------------------------
    input_df = pd.DataFrame([raw_input])

    # ----------------------------------------------
    # HANDLE MISSING COLUMNS
    # ----------------------------------------------
    for col in expected_col:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_col]

    # ----------------------------------------------
    # SCALE INPUT
    # ----------------------------------------------
    scaled_inp = scaler.transform(input_df)

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------
    pred = model.predict(scaled_inp)[0]

    # ----------------------------------------------
    # PROBABILITY
    # ----------------------------------------------
    prob = model.predict_proba(scaled_inp)[0][1]

    st.divider()

    # ----------------------------------------------
    # RESULT SECTION
    # ----------------------------------------------
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------
    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            label="Heart Disease Probability",
            value=f"{prob * 100:.2f}%"
        )

    with metric2:
        st.metric(
            label="Model Confidence",
            value=f"{max(prob, 1 - prob) * 100:.2f}%"
        )

    # ----------------------------------------------
    # GAUGE CHART
    # ----------------------------------------------
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': 'green'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ],
            }
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------
    # PATIENT SUMMARY
    # ----------------------------------------------
    st.subheader("🧾 Patient Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "Chest Pain",
            "Blood Pressure",
            "Cholesterol",
            "Max Heart Rate"
        ],
        "Value": [
            age,
            sex,
            pain,
            resting_bp,
            cholesterol,
            max_hr
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    # ----------------------------------------------
    # HEALTH TIPS
    # ----------------------------------------------
    st.subheader("💡 Health Recommendations")

    if pred == 1:
        st.warning(
            """
            - Maintain healthy diet
            - Exercise regularly
            - Avoid smoking
            - Reduce stress
            - Consult a cardiologist
            """
        )
    else:
        st.success(
            """
            - Continue healthy lifestyle
            - Maintain balanced diet
            - Exercise regularly
            - Keep monitoring health
            """
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    """
    <center>
        <h4>Made with ❤️ using Streamlit & Machine Learning</h4>
    </center>
    """,
    unsafe_allow_html=True
)