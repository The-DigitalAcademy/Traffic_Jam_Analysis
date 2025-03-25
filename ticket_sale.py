import streamlit as st
import joblib
import pandas as pd


model = joblib.load('random_forest_model.joblib')

expected_features = ['travel_from', 'Day', 'Minutes', 'car_type', 'max_capacity', 'Month', 'Hour', 'Year']

# Define a function to get user inputs
def get_user_input():
    
    travel_from = st.selectbox("Where is the trip starting from?", options=['Migori', 'Keroka', 'Homa Bay', 'Kisii', 'Keumbu', 'Rongo',
       'Kijauri', 'Oyugis', 'Awendo', 'Sirare', 'Nyachenge', 'Kehancha',
       'Kendu Bay', 'Sori', 'Rodi', 'Mbita', 'Ndhiwa'])
    car_type = st.selectbox("Type of car", options=['Bus', 'shuttle'])
    max_capacity = st.number_input("Max capacity of the car", min_value=1, max_value=10, value=5)
    day = st.number_input("Day of the trip (1-31)", min_value=1, max_value=31, value=15)
    minutes = st.number_input("Minutes of travel", min_value=0, max_value=300, value=60)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
    hour = st.number_input("Hour of the day (0-23)", min_value=0, max_value=23, value=12)
    year = st.number_input("Year of trip", min_value=2020, max_value=2025, value=2022)
    
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

# Streamlit app layout
st.title("Car Trip Prediction App")
st.write("Input the values for prediction:")

# Get user inputs
user_input = get_user_input()

# Display the input values
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