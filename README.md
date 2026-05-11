# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
Problem Statement

To develop a Random Forest Regression Model for predicting daily temperature, PM2.5 pollution level, and energy (TSR) using environmental sensor data collected from a weather monitoring station. The model analyzes sensor parameters such as humidity, CO₂, illumination, pressure, wind speed, and PM10 levels to make accurate predictions.

Dataset

Dataset Name: weather-station-eee-block_2024_07_13.csv

Input Features (Independent Variables)
hum → Humidity
co2 → Carbon dioxide level
illumination → Light intensity
pressure → Atmospheric pressure
pm10 → PM10 pollution level
wind_direction_angle → Wind direction angle
wind_speed → Wind speed
wind_speed_level → Wind speed category
Target Variables (Dependent Variables)
tem → Temperature
pm2_5 → PM2.5 pollution level
tsr → Energy / Solar Radiation value
Dataset Source

Environmental sensor readings collected from a weather station monitoring system.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset
2. Preprocess the Data
3. Train the Random Forest Regressor
4. Predict and Evaluate the Results

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: ISHWARYA M
RegisterNumber:  212225230107
*/
# Random Forest Algorithm for Predicting Temperature, PM2.5 and Energy

# Import required libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("weather-station-eee-block_2024_07_13.csv")

# Display dataset
print("Dataset Preview:")
print(data.head())

# Remove unwanted columns
data = data.drop(['time', 'wind_direction', 'bat'], axis=1)

# Fill missing values if any
data = data.fillna(data.mean(numeric_only=True))

# Input Features
X = data[['hum', 'co2', 'illumination', 'pressure',
          'pm10', 'wind_direction_angle',
          'wind_speed', 'wind_speed_level']]

# Target Variables
y = data[['tem', 'pm2_5', 'tsr']]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest Regressor model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance:")

print("Mean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# Predict for new sensor data
new_data = [[
    90,      # hum
    300,     # co2
    50000,   # illumination
    1000,    # pressure
    5,       # pm10
    180,     # wind_direction_angle
    1.5,     # wind_speed
    1        # wind_speed_level
]]

prediction = model.predict(new_data)

print("\nPredicted Values:")
print("Temperature:", prediction[0][0])
print("PM2.5:", prediction[0][1])
print("Energy (TSR):", prediction[0][2])
```

## Output:
<img width="827" height="728" alt="image" src="https://github.com/user-attachments/assets/3edc2f5c-72a8-4ef2-bac1-8cf5ec66f0d6" />


## Result:
The Random Forest Regression model successfully predicted daily temperature, PM2.5 pollution level, and energy values from environmental sensor data with good accuracy.
