import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load pre-trained churn model
with open('models/churn_model.pkl', 'rb') as file:
    churn_model = pickle.load(file)

def process_churn_data(filepath):
    # Load the customer data
    df = pd.read_csv(filepath)

    # Preprocess the data (scaling, encoding, etc.)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Predict churn using the pre-trained model
    predictions = churn_model.predict(df_scaled)

    # Add predictions to the DataFrame
    df['Churn Prediction'] = ['At Risk' if p == 1 else 'Not At Risk' for p in predictions]

    return df[['CustomerID', 'Churn Prediction']], df_scaled
