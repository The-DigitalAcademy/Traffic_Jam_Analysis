import streamlit as st
import pandas as pd
import numpy as np
import datetime
from joblib import load


model = load('random_forest_model.joblib')
travel_from_encoder = load('travel_from_encoder.joblib')
car_type_encoder = load('car_type_encoder.joblib')
mean = np.load('mean.npy')
std_dev = np.load('std_dev.npy')


def preprocess_data(df):
    df['travel_date'] = pd.to_datetime(df['travel_date'], format='%d-%m-%y')
    df['Year'] = df['travel_date'].dt.year
    df['Month'] = df['travel_date'].dt.month
    df['Day'] = df['travel_date'].dt.day
    df['Hour'] = pd.to_datetime(df['travel_time'], format='%H:%M').dt.hour
    df['Minutes'] = pd.to_datetime(df['travel_time'], format='%H:%M').dt.minute

    df['travel_from'] = travel_from_encoder.transform(df['travel_from'])
    df['car_type'] = car_type_encoder.transform(df['car_type'])

    return df


def standardize_data(X):
    return (X - mean) / std_dev


st.title("Ticket Sales Prediction")
st.subheader("Enter details to predict ticket sales")


travel_from_options = travel_from_encoder.classes_
car_type_options = car_type_encoder.classes_
date_options = [f"{day:02d}-03-25" for day in range(1, 32)]
time_options = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in [0, 30]]


travel_from = st.selectbox("Travel From (Location)", travel_from_options)
car_type = st.selectbox("Car Type", car_type_options)
max_capacity = st.number_input("Max Capacity", min_value=1, step=1)

travel_date = st.date_input(
    "Select Travel Date",
    value=datetime.date.today(),  
)

travel_time = st.text_input(
    "Enter Travel Time (HH:MM)",
    value="12:00",  
)

if st.button("Predict"):
    if travel_from and car_type and travel_date and travel_time:
        input_data = pd.DataFrame({
            "travel_from": [travel_from],
            "car_type": [car_type],
            "max_capacity": [max_capacity],
            "travel_date": [travel_date],
            "travel_time": [travel_time]
        })

        processed_data = preprocess_data(input_data)
        X = processed_data[['travel_from', 'car_type', 'max_capacity', 'Year', 'Month', 'Day', 'Hour', 'Minutes']]
        X = standardize_data(X)

        prediction = model.predict(X)
        st.success(f"Predicted Number of Tickets: {prediction[0]}")
    else:
        st.error("Please fill in all fields.")
