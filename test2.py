import os
import csv
import pandas as pd
from termcolor import colored as cl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def preprocess_input(X_custom):
    # Load your dataset into a Pandas DataFrame
    data = pd.read_csv('Capstone/data.csv')

    # Create a new DataFrame with modified features
    modified_data = data.copy()  # Create a copy to avoid modifying the original DataFrame
    modified_data['Parking Spaces'] = data['Carport Spaces'] + data['Garage Spaces']

    # Select relevant features for preprocessing
    selected_features = ["Baths", "Parking Spaces", "Lot Size SqFt", "Fin SqFt Total"]

    # Define features (X) using the modified DataFrame
    X = modified_data[selected_features]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the custom input
    X_custom_scaled = scaler.transform(X_custom)

    return X_scaled, X_custom_scaled

def make_prediction(X_custom):
    try:
        # Preprocess input data
        X_scaled, X_custom_scaled = preprocess_input(X_custom)

        # Load your dataset into a Pandas DataFrame
        data = pd.read_csv('Capstone/data.csv')

        # Define target variable (y)
        y = data['Price Sold']  # Use the target variable from the original DataFrame

        # Split the dataset into training and testing sets
        _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize and fit the multilinear regression model
        regressor = LinearRegression(positive=True)
        regressor.fit(X_scaled, y)

        # Save the trained model to a file
        model_path = '/home/yhs/Documents/Scripts/Capstone/trained_model.pkl'
        joblib.dump(regressor, model_path)

        # Predict the sales prices for custom input
        y_pred_custom = regressor.predict(X_custom_scaled)

        return y_pred_custom
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    # Call the make_prediction function
    bath = 3
    parking = 5
    lot_size = 7200
    finished_area = 2100
    X_custom = np.array([[bath, parking, lot_size, finished_area]])
    predicted_price = make_prediction(X_custom)
    print("Predicted Price:", round(predicted_price[0]))
