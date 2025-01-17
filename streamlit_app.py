import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import great_circle 
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib

class LogTransfomer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):  # always return self
        # calculate what is needed to make .transform()
        # self.mean_ = np.mean(x)
        self.n_features_in_ = x.shape[1] 
        return self # always return self
    
    def transform(self, x, y=None):
        assert self.n_features_in_ == x.shape[1]
        return np.log1p(x)

# load model
lr_grid = joblib.load("lr_grid.pkl")

# helper function
def get_season(arg): # month=1
    if arg in [12, 1, 2]:
        return "Winter"
    elif arg in [3, 4, 5]:
        return "Spring"
    elif arg in [6, 7, 8]:
        return "Summer"
    else: # 9 10 11
        return "Autumn"

def get_dayperiod(arg): # hour=1
    if arg > 5 and arg < 12:
        return "Morning"
    elif arg >= 12 and arg < 17:
        return "Afternoon"
    elif arg >= 17 and arg < 22:
        return "Evenning"
    else: # 22 -> 5
        return "Night"
        
def get_distance(location_1, location_2):
    # location_1 = ("40.738354", "-73.999817") # (lat, lon)
    # location_2 = ("40.723217", "-73.999512") # (lat, lat)
    return great_circle(location_1, location_2).km

# Welcome message     
st.title("Welcome to Uber Fare Prediction")
st.text("information about dataset")

# inputs
pickup_lat = float(st.number_input("put pickup latitude value: ", min_value=-90.0, max_value=90.0, format='%.20f'))
pickup_lon = float(st.number_input("put pickup longitude value: ", min_value=-180.0, max_value=180.0, format='%.20f'))
dropoff_lat = float(st.number_input("put dropoff latitude value: ", min_value=-90.0, max_value=90.0, format='%.20f'))
dropoff_lon = float(st.number_input("put dropoff longitude value: ", min_value=-180.0, max_value=180.0, format='%.20f'))
date = st.date_input("Please put date where he picked uber: ", min_value=datetime(2009, 1, 1), max_value=datetime.now())
hour = int(st.number_input("Please pick hour where he picked uber: ", min_value=0, max_value=23))
passenger_count = int(st.slider("choose number of passengers: ", min_value=1, max_value=6))
picked_datetime = datetime(date.year, date.month, date.day, hour)

# predict
predict = st.button("Predict")
if predict:
    print(f"pickup_lat: {pickup_lat}")
    print(f"pickup_lon: {pickup_lon}")
    print(f"dropoff_lat: {dropoff_lat}")
    print(f"dropoff_lon: {dropoff_lon}")
    print(f"passenger_count: {passenger_count}")
    print(f"picked_datetime: {picked_datetime}")
    pickup_year = picked_datetime.year
    pickup_month = picked_datetime.month
    pickup_weekday = picked_datetime.weekday()
    pickup_hour = picked_datetime.hour

    pickup_season = get_season(pickup_month)
    pickup_period = get_dayperiod(pickup_hour)
    location_1 = (pickup_lat, pickup_lon)
    location_2 = (dropoff_lat, dropoff_lon)

    distance = get_distance(location_1, location_2)
    columns = ['passenger_count', 'pickup_year', 'pickup_month', 'pickup_weekday', 'pickup_hour', 'pickup_season', 'pickup_period', 'distance']
    data = pd.DataFrame([[passenger_count, pickup_year, pickup_month, pickup_weekday, pickup_hour, pickup_season, pickup_period, distance]], columns=columns)
    st.dataframe(data)

    # prediction
    with st.spinner():
        time.sleep(5)
        pred = 'Please enter valid distance'
        if distance != 0:
            pred = np.exp(lr_grid.predict(data).ravel()[0])
        
    st.text(f"Uber Fare: {pred}")
