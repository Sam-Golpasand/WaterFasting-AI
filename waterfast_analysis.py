import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


file_path = 'testData.xlsx' 
sheet_name = 'Ark1' 

# Define expected columns
expected_columns = [
    'patientnumber', 'length', 'weightpre', 'weightpost', 
    'bmipre', 'bmipost', 'waistpre', 'waistpost', 
    'pulsepre', 'pulsepost'
]

# Load and preprocess the dataset
try:
    # Load the Excel file
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Dataset loaded successfully!")

    # Normalize column names to lowercase
    data.columns = data.columns.str.lower()

    # Debug: Show the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())

    # Check if all required columns are present
    missing_cols = [col for col in expected_columns if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in dataset: {missing_cols}")

    # Select relevant columns and ensure numeric types
    data_cleaned = data[expected_columns].copy()
    data_cleaned = data_cleaned.apply(pd.to_numeric, errors='coerce').fillna(data.median(numeric_only=True))

    # Drop non-numerical or irrelevant columns for anomaly detection
    numerical_cols = expected_columns[1:]  # Exclude 'patient number'
    data_scaled = StandardScaler().fit_transform(data_cleaned[numerical_cols])
    
    print("Data preprocessed successfully!")
except Exception as e:
    print(f"Error during data preprocessing: {e}")
    exit(1)


# Train the Isolation Forest model
try:
    model = IsolationForest(contamination=0.17, random_state=42)
    model.fit(data_scaled)
    print("Isolation Forest model trained successfully!")
except Exception as e:
    print(f"Error during model training: {e}")
    exit(1)

def assess_data(new_data_path, sheet_name='Ark1'):
    try:
        # Load new dataset 
        new_data = pd.read_excel(new_data_path, sheet_name=sheet_name)
        
        # Normalize column names to lowercase in the new data 
        new_data.columns = new_data.columns.str.lower()

        #print("Columns in the new dataset:", new_data.columns)

        # Check for required columns
        missing_cols = [col for col in expected_columns if col not in new_data.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in new dataset: {missing_cols}")

        # Preprocess new data
        new_data_cleaned = new_data[expected_columns].copy()
        new_data_cleaned = new_data_cleaned.apply(pd.to_numeric, errors='coerce').fillna(data_cleaned.median())
        new_data_scaled = StandardScaler().fit_transform(new_data_cleaned[numerical_cols])

        # Predict anomalies (1 = normal, -1 = anomaly)
        predictions = model.predict(new_data_scaled)
        results = pd.DataFrame({
            'Patient': new_data['patientnumber'],  # Adjust to correct column name
            'Anomaly': predictions
        })
        return results
    except Exception as e:
        print(f"Error during data assessment: {e}")
        return None


if __name__ == "__main__":
    try:
        new_data_path = 'testData.xlsx'
        results = assess_data(new_data_path)

        if results is not None:
            print("Anomaly detection results:")
            print(results)
            results.to_csv('oldResultsData.csv', index=False)
            print("Results saved to 'oldResultsData.csv'.")
    except Exception as e:
        print(f"Unexpected error: {e}")
