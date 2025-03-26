import streamlit as st
import joblib
import pandas as pd


model = joblib.load('random_forest_model.joblib')

expected_features = ['travel_from', 'Day', 'Minutes', 'car_type', 'max_capacity', 'Month', 'Hour', 'Year']


def get_user_input():
    
    travel_from = st.selectbox("Where is the trip starting from?", options=['Migori', 'Keroka', 'Homa Bay', 'Kisii', 'Keumbu', 'Rongo',
       'Kijauri', 'Oyugis', 'Awendo', 'Sirare', 'Nyachenge', 'Kehancha',
       'Kendu Bay', 'Sori', 'Rodi', 'Mbita', 'Ndhiwa'])
    car_type = st.selectbox("Type of car", options=['Bus', 'shuttle'])
    max_capacity = st.number_input("Max capacity of the car")
    day = st.number_input("Day of the trip")
    minutes = st.number_input("Minutes of travel")
    month = st.number_input("Month")
    hour = st.number_input("Hour of the day")
    year = st.number_input("Year of trip")
    
    user_data = {
        'travel_from': travel_from,
        'car_type': car_type,
        'max_capacity': max_capacity,
        'Day': day,
        'Minutes': minutes,
        'Month': month,
        'Hour': hour,
        'Year': year
    }

    return pd.DataFrame([user_data])[expected_features]


st.title("Ticket Sale Prediction")



user_input = get_user_input()


st.write("Your Input Data:")
st.write(user_input)

# Predict using the trained model
if st.button("Get Prediction"):
    try:
        # Ensure the input features match the training order
        prediction = model.predict(user_input)
        st.write("Prediction Result:", prediction)
    except ValueError as e:
        st.error(f"Error: {e}")