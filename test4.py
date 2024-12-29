import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import tkinter as tk

def preprocess_input(X_custom):
    # Load your dataset into a Pandas DataFrame
    data = pd.read_csv('Capstone/data.csv')

    # Create a new DataFrame with modified features
    modified_data = data.copy()  # Create a copy to avoid modifying the original DataFrame
    modified_data['Parking Spaces'] = data['Carport Spaces'] + data['Garage Spaces']

    # Select relevant features for preprocessing
    selected_features = ["Baths", "Parking Spaces", "Lot Size SqFt", "SqFt Fin Total"]

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

def submit():
    # Get user input from entry fields
    bath = int(bath_entry.get())
    parking = int(parking_entry.get())
    lot_size = int(lot_size_entry.get())
    finished_area = int(finished_area_entry.get())
    
    # Prepare custom input
    X_custom = np.array([[bath, parking, lot_size, finished_area]])
    
    # Make prediction
    predicted_price = load_model_and_make_prediction(X_custom)
    
    # Display predicted price below the input fields
    prediction_label.config(text=f"Predicted Price: {round(predicted_price[0], 2)}", font=("Helvetica", 48))

# Create main tkinter window
root = tk.Tk()
root.title("House Price Prediction")

# Create labels and entry fields for user input
tk.Label(root, text="Bathrooms:", font=("Helvetica", 48)).grid(row=0, column=0)
bath_entry = tk.Entry(root, font=("Helvetica", 48))
bath_entry.grid(row=0, column=1)

tk.Label(root, text="Parking Spaces:", font=("Helvetica", 48)).grid(row=1, column=0)
parking_entry = tk.Entry(root, font=("Helvetica", 48))
parking_entry.grid(row=1, column=1)

tk.Label(root, text="Lot Size (SqFt):", font=("Helvetica", 48)).grid(row=2, column=0)
lot_size_entry = tk.Entry(root, font=("Helvetica", 48))
lot_size_entry.grid(row=2, column=1)

tk.Label(root, text="Finished Area (SqFt):", font=("Helvetica", 48)).grid(row=3, column=0)
finished_area_entry = tk.Entry(root, font=("Helvetica", 48))
finished_area_entry.grid(row=3, column=1)

# Create submit button
submit_button = tk.Button(root, text="Submit", command=submit, font=("Helvetica", 48))
submit_button.grid(row=4, column=0, columnspan=2)

# Create label for predicted price
prediction_label = tk.Label(root, font=("Helvetica", 48))
prediction_label.grid(row=5, column=0, columnspan=2)

# Run the tkinter event loop
root.mainloop()
