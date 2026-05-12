import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_customers=1000, filename='customers.csv'):
    np.random.seed(42)
    random.seed(42)

    data = {
        'CustomerID': [f'C{str(i).zfill(4)}' for i in range(1, num_customers + 1)],
        'Age': np.random.randint(18, 70, size=num_customers),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], size=num_customers, p=[0.48, 0.48, 0.04]),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_customers, p=[0.5, 0.3, 0.2]),
        'Income': np.random.normal(60000, 20000, size=num_customers).clip(20000, 150000).astype(int),
        'PurchaseFrequency': np.random.poisson(lam=5, size=num_customers), # Number of purchases per year
        'Recency': np.random.randint(1, 365, size=num_customers), # Days since last purchase
    }
    
    # Generate spending correlated with income and frequency
    data['TotalSpend'] = (data['Income'] * 0.01 * data['PurchaseFrequency'] * np.random.uniform(0.5, 1.5, size=num_customers)).astype(float)
    
    # Avoid 0 spend if they have purchases, and if frequency is 0, spend is 0
    data['TotalSpend'] = np.where(data['PurchaseFrequency'] == 0, 0, data['TotalSpend'])
    # Fix recency if they never purchased
    data['Recency'] = np.where(data['PurchaseFrequency'] == 0, 999, data['Recency'])
    
    data['WebsiteVisits'] = data['PurchaseFrequency'] * np.random.randint(1, 5, size=num_customers) + np.random.randint(0, 10, size=num_customers)
    data['CampaignResponse'] = np.random.choice([0, 1], size=num_customers, p=[0.8, 0.2])

    df = pd.DataFrame(data)
    
    # Round float columns
    df['TotalSpend'] = df['TotalSpend'].round(2)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Generated synthetic data for {num_customers} customers and saved to {filename}")

if __name__ == "__main__":
    generate_synthetic_data()
