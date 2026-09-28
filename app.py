# 32. Streamlit Application
# هنا بنجهز كود التطبيق النهائي.
# التطبيق هيعمل الآتي:
#
# 1. يحمّل الـModel المحفوظ.
# 2. يحمّل الـScaler المحفوظ.
# 3. يقرأ بيانات الطقس.
# 4. يأخذ آخر 24 قراءة.
# 5. يعمل Scaling بنفس الـScaler المستخدم أثناء التدريب.
# 6. يمرر البيانات للـModel.
# 7. يرجع درجة الحرارة المتوقعة.
#
# مهم:
# لازم يكون عندنا:
# jena_climate_best_model.keras
# jena_climate_scaler.pkl
# jena_climate_2009_2016.csv


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Jena Climate AI",
    page_icon="🌤️",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🌤️ Jena Climate AI")
st.subheader("Temperature Forecasting with Deep Learning")

st.write(
    "This application uses a Deep Learning time-series model "
    "to predict the next temperature value based on the previous 24 readings."
)


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "jena_climate_best_model.keras"
    )


# -----------------------------
# Load Scaler
# -----------------------------

@st.cache_resource
def load_scaler():
    return joblib.load(
        "jena_climate_scaler.pkl"
    )


model = load_model()
scaler = load_scaler()


# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():

    data = pd.read_csv(
        "jena_climate_2009_2016.csv"
    )

    return data


df_app = load_data()


# -----------------------------
# Prepare Temperature Data
# -----------------------------

temperature_data = df_app[
    ["T (degC)"]
].copy()


# -----------------------------
# Get Last 24 Readings
# -----------------------------

sequence_length = 24

last_values = temperature_data.tail(
    sequence_length
).values


# -----------------------------
# Scaling
# -----------------------------

scaled_values = scaler.transform(
    last_values
)


# -----------------------------
# Reshape for Deep Learning Model
# -----------------------------

X_input = scaled_values.reshape(
    1,
    sequence_length,
    1
)


# -----------------------------
# Prediction Button
# -----------------------------

if st.button("🌡️ Predict Next Temperature"):

    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction_scaled
    )

    predicted_temperature = float(
        prediction[0][0]
    )


    # -----------------------------
    # Display Result
    # -----------------------------

    st.success(
        f"Predicted Next Temperature: "
        f"{predicted_temperature:.2f} °C"
    )


# -----------------------------
# Information
# -----------------------------

st.divider()

st.write(
    f"Model used: **{model.name}**"
)

st.write(
    f"Sequence length: **{sequence_length} readings**"
)