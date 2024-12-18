# Anomaly Detector

## Overview
The **Anomaly Detector** is a Python-based tool designed to identify anomalies in patient-related data using machine learning. It preprocesses data, trains models, and evaluates new datasets for potential anomalies.

---

## Requirements

- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - openpyxl (for reading Excel files)

Install dependencies using:
```bash
pip install pandas numpy scikit-learn openpyxl
```

---

## Structure

### Files
- **`trainingData.xlsx`**: Training dataset (default sheet name: `Ark1`).
- **`testData.xlsx`**: Test dataset to evaluate anomalies (default sheet name: `Ark1`).
- **`resultsData.csv`**: Output file containing anomaly detection results.


## How to Use

1. **Prepare Data**:
   - Ensure the training and test datasets have the following columns:
     - `patientnumber`, `length`, `weightpre`, `weightpost`, `bmipre`, `bmipost`, `waistpre`, `waistpost`, `pulsepre`, `pulsepost`.
   - Save them as `trainingData.xlsx` and `testData.xlsx` respectively.
   - 2 files have already been created with example data.

2. **Run the Script**:
   Execute the script:
   ```bash
   python main.py
   ```

3. **Output**:
   - The anomaly detection result will be displayed in the console.
   - Results are saved in `resultsData.csv`.

---

## Example Result

### Console Output
```plaintext
Anomaly Detection Result:
{
    'Patient': 1,
    'Isolation_Forest_Anomaly': True,
    'Distance_Based_Anomaly': False,
    'length_Z_Score': 0.5,
    'weightpre_Z_Score': 2.1,
    ...
}
Results saved to 'resultsData.csv'
```

### CSV Output
| Patient | Isolation_Forest_Anomaly | Distance_Based_Anomaly | length_Z_Score | weightpre_Z_Score | ... |
|---------|---------------------------|------------------------|----------------|-------------------|-----|
| 101     | True                      | False                 | 0.5            | 2.1               | ... |

---


## License
This project is open-source and available for use and modification.

