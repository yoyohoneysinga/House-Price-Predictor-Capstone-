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

def data_preprocessing(folder_path, keywords):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as txt_file:
                lines = txt_file.readlines()

            # Create a dictionary to map original headers to keywords
            header_mapping = {
                "Fin SqFt Total": "SqFt Fin Total",
                "Parking Total Spaces": "Parking Spaces",
                "Basement?": "Basement",
            }

            # Process each line to replace headers with keywords
            updated_lines = []
            for line in lines:
                for original_header, keyword in header_mapping.items():
                    if original_header in line:
                        print(f"Changed '{original_header}' to '{keyword}' in file: {filename}")
                        line = line.replace(original_header, keyword)
                updated_lines.append(line)

            # Write updated lines to the same text file
            with open(os.path.join(folder_path, filename), 'w') as txt_file:
                txt_file.writelines(updated_lines)

"""
Example Usage:
folder_path = 'Capstone/HouseEntries'
keywords = ["Beds", "Baths", "Basement", "Parking Spaces", "Garage Spaces",
            "Carport Spaces", "Lot Size SqFt", "Price Sold", "Kitchens",
            "Fireplaces Total", "Year Built (Est)", "Fin SqFt Total",
            "SqFt Unfinished", "Lot Size Acres", "Sub area"]
data_preprocessing(folder_path, keywords)
"""

def extract_data_to_csv(folder_path, output_csv_path, keywords):
    # Open the output CSV file in append mode
    with open(output_csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Check if the output CSV file is empty and write the headers if necessary
        if os.path.getsize(output_csv_path) == 0:
            csv_writer.writerow(keywords + ['Filename'])  # Add 'Filename' as an additional header

        # Iterate over each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r') as txt_file:
                    lines = txt_file.readlines()
                    data = {}

                    # Extract relevant data from the text file
                    for line in lines:
                        for keyword in keywords:
                            if keyword in line:
                                parts = line.strip().split('|')
                                if len(parts) == 2:
                                    key = parts[0].strip()
                                    value = parts[1].strip()
                                    data[key] = value

                    # Write data to the CSV file
                    row_data = [data.get(keyword, '') for keyword in keywords]
                    row_data.append(os.path.splitext(filename)[0])  # Add filename without extension
                    csv_writer.writerow(row_data)

"""
Extract Data to CSV:
folder_path = 'Capstone/HouseEntries'
output_csv_path = 'Capstone/data.csv'
keywords = ["Beds", "Baths", "Basement", "Parking Spaces", "Garage Spaces",
            "Carport Spaces", "Lot Size SqFt", "Price Sold", "Kitchens",
            "Fireplaces Total", "Year Built (Est)", "SqFt Fin Total",
            "SqFt Unfinished", "Lot Size Acres", "Sub area"]
extract_data_to_csv(folder_path, output_csv_path, keywords)
"""

def preprocess_and_update_csv(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Print the number of missing values for each column
    print(cl(df.isnull().sum(), attrs=['bold']))
    
    # Convert 'Year Built (Est)' to integer
    df['Year Built (Est)'] = pd.to_numeric(df['Year Built (Est)'], errors='coerce').astype('Int64')
    
    # Convert 'Lot Size Acres' to integer
    df['Lot Size Acres'] = pd.to_numeric(df['Lot Size Acres'], errors='coerce').astype('Int64')
    
    # Encode categorical variables using LabelEncoder
    le = LabelEncoder()
    df['Basement'] = le.fit_transform(df['Basement'])
    df['Sub area'] = le.fit_transform(df['Sub area'])
    
    # Convert string values with commas to integers
    df['Lot Size SqFt'] = df['Lot Size SqFt'].str.replace(',', '').astype(float).astype(int)
    df['SqFt Fin Total'] = df['SqFt Fin Total'].str.replace(',', '').astype(float).astype(int)
    df['SqFt Unfinished'] = df['SqFt Unfinished'].str.replace(',', '').astype(float).astype(int)
    
    # Convert 'Price Sold' to integer after removing special characters
    df['Price Sold'] = df['Price Sold'].replace('[\$,]', '', regex=True).astype(float).astype(int)
    
    # Overwrite the original CSV file with the updated data
    df.to_csv(csv_path, index=False)
    
    # Print the data types of each column after preprocessing
    print(cl(df.dtypes, attrs=['bold']))

"""
# Example usage:
csv_path = 'Capstone/data.csv'
preprocess_and_update_csv(csv_path)
"""

def lazypredict():
    # Load your dataset into a Pandas DataFrame
    data = pd.read_csv('Capstone/data.csv')

    # Define features (X) and target variable (y)
    X = data[["Beds", "Baths", "Basement", "Parking Spaces", "Garage Spaces",
            "Carport Spaces", "Lot Size SqFt", "Kitchens", "Fireplaces Total",
            "Year Built (Est)", "SqFt Fin Total", "SqFt Unfinished",
            "Lot Size Acres", "Sub area"]]
    y = data['Price Sold']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use LazyRegressor for regression
    reg = LazyRegressor(predictions=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Print the performance of various regression models
    return(models)

"""
# Example usage:
print(lazypredict())
"""

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

def make_prediction(X_custom):
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

    # Predict the sales prices for custom input
    y_pred_custom = regressor.predict(X_custom_scaled)

    return y_pred_custom

# Example usage:
bath = 3
parking = 5
lot_size = 7200
finished_area = 2100
X_custom = np.array([[bath, parking, lot_size, finished_area]])  # Custom input with custom parameters for each feature
predicted_price = make_prediction(X_custom)
print("Predicted Price:", round(predicted_price[0]))

def multilinear_regression_with_known_coefficient(known_coefficient_value):
    # Load your dataset into a Pandas DataFrame
    data = pd.read_csv('Capstone/data.csv')

    # Define features (X) and target variable (y)
    X = data[["Baths", "Parking Spaces", "Lot Size SqFt",
              "Year Built (Est)", "SqFt Fin Total", "SqFt Unfinished"]]
    y = data['Price Sold']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the multilinear regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict the sales prices
    y_pred = regressor.predict(X_test)

    # Adjust the predictions based on the known coefficient
    adjusted_predictions = y_pred + known_coefficient_value

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(cl(f"Mean Absolute Error: {mae}", attrs=['bold']))
    print(cl(f"Mean Squared Error: {mse}", attrs=['bold']))
    print(cl(f"R-squared: {r2}", attrs=['bold']))

    # Print the coefficients for each variable
    coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': regressor.coef_})
    print(cl(coefficients, attrs=['bold']))

    # Return the adjusted predictions
    return adjusted_predictions

"""
# Example usage:
known_coefficient_value = 304  # Replace with the known coefficient value
adjusted_predictions = multilinear_regression_with_known_coefficient(known_coefficient_value)
print("Adjusted Predictions:", adjusted_predictions)
"""

