import joblib
import numpy as np
import pandas as pd
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

def load_model_and_make_prediction(X_custom):
    # Load the trained model
    regressor = joblib.load('/home/yhs/Documents/Scripts/Capstone/trained_model.pkl')

    # Preprocess the custom input data
    X_scaled, X_custom_scaled = preprocess_input(X_custom)

    # Predict the sales prices for custom input
    y_pred_custom = regressor.predict(X_custom_scaled)

    return y_pred_custom

# Example usage:
bath = 3
parking = 5
lot_size = 7200
finished_area = 2100
X_custom = np.array([[bath, parking, lot_size, finished_area]])  # Custom input with custom parameters for each feature
predicted_price = load_model_and_make_prediction(X_custom)
print("Predicted Price:", round(predicted_price[0]))
