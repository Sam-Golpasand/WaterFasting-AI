import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, training_data_path, training_sheet_name='Ark1'):
        # Define expected columns
        self.expected_columns = [
            'patientnumber', 'length', 'weightpre', 'weightpost',
            'bmipre', 'bmipost', 'waistpre', 'waistpost',
            'pulsepre', 'pulsepost'
        ]
        # Load and preprocess training data
        self.training_data = self.load_and_preprocess_data(training_data_path, training_sheet_name)
        # Train models
        self.train_models()

    def load_and_preprocess_data(self, file_path, sheet_name):
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        data.columns = data.columns.str.lower()
        missing_cols = [col for col in self.expected_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        data_cleaned = data[self.expected_columns].copy()
        for col in data_cleaned.columns:
            data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
        data_cleaned.fillna(data_cleaned.median(), inplace=True)
        return data_cleaned

    def train_models(self):
        numerical_cols = [col for col in self.expected_columns if col != 'patientnumber']
        self.scaler = RobustScaler()
        training_scaled = self.scaler.fit_transform(self.training_data[numerical_cols])
        
        self.isolation_forest = IsolationForest(contamination=0.23, random_state=42)
        self.isolation_forest.fit(training_scaled)
        
        self.feature_stats = {
            col: {'mean': self.training_data[col].mean(), 'std': self.training_data[col].std()}
            for col in numerical_cols
        }

    def assess_new_data(self, new_data_path, new_sheet_name='Ark1'):
        new_data = pd.read_excel(new_data_path, sheet_name=new_sheet_name)
        new_data.columns = new_data.columns.str.lower()
        
        missing_cols = [col for col in self.expected_columns if col not in new_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in new dataset: {missing_cols}")
        
        # Take the first row only
        new_data_cleaned = new_data.head(1)[self.expected_columns].copy()
        for col in new_data_cleaned.columns:
            new_data_cleaned[col] = pd.to_numeric(new_data_cleaned[col], errors='coerce')
        new_data_cleaned.fillna(
            {col: self.training_data[col].median() for col in new_data_cleaned.columns}, 
            inplace=True
        )
        
        numerical_cols = [col for col in self.expected_columns if col != 'patientnumber']
        new_data_scaled = self.scaler.transform(new_data_cleaned[numerical_cols])

        # Isolation Forest prediction
        isolation_score = self.isolation_forest.predict(new_data_scaled)[0]
        
        # Distance-based anomaly detection
        distance_anomalies = [
            abs(new_data_cleaned.iloc[0][col] - self.feature_stats[col]['mean']) / self.feature_stats[col]['std'] > 2
            for col in numerical_cols
        ]
        distance_based_anomaly = any(distance_anomalies)
        
        # Z-scores for features
        feature_z_scores = {
            f'{col}_Z_Score': abs(new_data_cleaned.iloc[0][col] - self.feature_stats[col]['mean']) / self.feature_stats[col]['std']
            for col in numerical_cols
        }
        
        # Combine results
        result = {
            'Patient': new_data_cleaned['patientnumber'].iloc[0],
            'Isolation_Forest_Anomaly': isolation_score == -1,
            'Distance_Based_Anomaly': distance_based_anomaly,
            **feature_z_scores
        }
        return result

def main():
    try:
        training_data_path = 'trainingData.xlsx'
        test_data_path = 'testData.xlsx'
        
        detector = AnomalyDetector(training_data_path)
        result = detector.assess_new_data(test_data_path)
        
        # Display result
        print("Anomaly Detection Result:")
        print(result)
        
        # Save result as CSV
        result_df = pd.DataFrame([result])
        result_df.to_csv('resultsData.csv', index=False)
        print("Results saved to 'resultsData.csv'")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
