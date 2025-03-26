import streamlit as st
import pandas as pd
from joblib import load


model = load('random_forest_model.joblib')


st.title("Ticket Sales Prediction")


st.sidebar.header("Enter the Details Below")


def get_user_input():
 
    travel_from_options = {
        'Migori': 0, 'Keroka': 1, 'Homa Bay': 2, 'Kisii': 3, 'Keumbu': 4,
        'Rongo': 5, 'Kijauri': 6, 'Oyugis': 7, 'Awendo': 8, 'Sirare': 9,
        'Nyachenge': 10, 'Kehancha': 11, 'Kendu Bay': 12, 'Sori': 13,
        'Rodi': 14, 'Mbita': 15, 'Ndhiwa': 16
    }


    car_type_options = {'Bus': 0, 'Shuttle': 1}

  
    travel_from = st.sidebar.selectbox("Travel From", options=list(travel_from_options.keys())) 
    car_type = st.sidebar.selectbox("Car Type", options=list(car_type_options.keys())) 
    max_capacity = st.sidebar.slider("Max Capacity", 10, 100, 50)  
    year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2025)  
    month = st.sidebar.slider("Month", 1, 12, 1) 
    day = st.sidebar.slider("Day", 1, 31, 1) 
    hour = st.sidebar.slider("Hour", 0, 23, 0) 
    minutes = st.sidebar.slider("Minutes", 0, 59, 0)  


    display_data = {
        'Travel From': travel_from,
        'Car Type': car_type,
        'Max Capacity': max_capacity,
        'Year': year,
        'Month': month,
        'Day': day,
        'Hour': hour,
        'Minutes': minutes
    }


    encoded_data = {
        'travel_from': travel_from_options[travel_from],
        'car_type': car_type_options[car_type],
        'max_capacity': max_capacity,
        'Year': year,
        'Month': month,
        'Day': day,
        'Hour': hour,
        'Minutes': minutes
    }

    return pd.DataFrame(display_data, index=[0]), pd.DataFrame(encoded_data, index=[0])


display_data, encoded_data = get_user_input()

# st.write("Your Input:")
# st.write(display_data)

prediction = model.predict(encoded_data)

st.write(f"Predicted Number of Tickets Sold: {int(prediction[0])}")
