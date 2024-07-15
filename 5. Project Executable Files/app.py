from flask import Flask, render_template, request, url_for
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/response', methods=['Post'])
def response():
    dt = '2016-01-01'
    LandAverageTemperatureUncertainty = float(request.form["LandAverageTemperatureUncertainty"])
    LandMaxTemperature = float(request.form["LandMaxTemperature"])
    LandMaxTemperatureUncertainty = float(request.form["LandMaxTemperatureUncertainty"])
    LandMinTemperature = float(request.form["LandMinTemperature"])
    LandMinTemperatureUncertainty = float(request.form["LandMinTemperatureUncertainty"])
    LandAndOceanAverageTemperature = float(request.form["LandAndOceanAverageTemperature"])
    LandAndOceanAverageTemperatureUncertainty = float(request.form["LandAndOceanAverageTemperatureUncertainty"])

    print("\n")
    print(f"dt: {dt}, type: {type(dt)}")
    print(f"LandAverageTemperatureUncertainty: {LandAverageTemperatureUncertainty}, type: {type(LandAverageTemperatureUncertainty)}")
    print(f"LandMaxTemperature: {LandMaxTemperature}, type: {type(LandMaxTemperature)}")
    print(f"LandMaxTemperatureUncertainty: {LandMaxTemperatureUncertainty}, type: {type(LandMaxTemperatureUncertainty)}")
    print(f"LandMinTemperature: {LandMinTemperature}, type: {type(LandMinTemperature)}")
    print(f"LandMinTemperatureUncertainty: {LandMinTemperatureUncertainty}, type: {type(LandMinTemperatureUncertainty)}")
    print(f"LandAndOceanAverageTemperature: {LandAndOceanAverageTemperature}, type: {type(LandAndOceanAverageTemperature)}")
    print(f"LandAndOceanAverageTemperatureUncertainty: {LandAndOceanAverageTemperatureUncertainty}, type: {type(LandAndOceanAverageTemperatureUncertainty)}")

    # Sample input data
    input_data = [dt, LandAverageTemperatureUncertainty, LandMaxTemperature, LandMaxTemperatureUncertainty, 
                LandMinTemperature, LandMinTemperatureUncertainty, LandAndOceanAverageTemperature, 
                LandAndOceanAverageTemperatureUncertainty]

    # Extract year and month from the date string
    input_date = pd.to_datetime(input_data[0])  # Convert date string to datetime object
    year = input_date.year  # Extract year from datetime object
    month = input_date.month  # Extract month from datetime object

    # Prepare the input array with year, month, and other features
    input_array = [year, month] + input_data[1:]

    # Convert the input array to a DataFrame for scaling
    input_df = pd.DataFrame([input_array], columns=['LandAverageTemperatureUncertainty', 
                                                    'LandMaxTemperature', 'LandMaxTemperatureUncertainty', 
                                                    'LandMinTemperature', 'LandMinTemperatureUncertainty', 
                                                    'LandAndOceanAverageTemperature', 
                                                    'LandAndOceanAverageTemperatureUncertainty','Year', 'Month'])

    # Load the scaler
    scaler_x = joblib.load('scaler_x.pkl')

    # Apply the previously defined scaler_x to transform the input DataFrame
    input_scaled = scaler_x.transform(input_df)

    # Reshape the input to match the LSTM model's expected input shape (samples, timesteps, features)
    input_reshaped = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))

    # Print the shape of the reshaped input array for verification
    print("Input reshaped:", input_reshaped.shape)

    # Load the trained model from the specified file ('best_model.keras')
    loaded_model = load_model('best_model.keras')

    # Predict using the loaded model on the reshaped input data
    prediction = loaded_model.predict(input_reshaped)
    print("Scaled prediction:", prediction)

    # Print the shape of the prediction array to verify its dimensions
    print("Prediction shape:", prediction.shape)

    # Load the scaler_y
    scaler_y = joblib.load('scaler_y.pkl')

    # Inverse transform the predicted values to get the original scale of 'y'
    prediction_inverse = scaler_y.inverse_transform(prediction)

    # Print the first predicted value after inverse transformation
    print("Inverse transformed prediction:", prediction_inverse[0][0])

    float_value = prediction_inverse[0][0]


    return render_template('index.html',T=dt,S=float_value)

if __name__ == '__main__':
    app.run()


'''
Authors:
- OZA ASHWIN
- RUTHWIK SAI
- ANUSHA T
- RAGHAVVRAM J
'''