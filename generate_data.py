# generate_data.py
import pandas as pd
import numpy as np
import os

def generate_sample_data():
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    regions = ['North', 'South', 'East', 'West']

    data = []
    for date in dates:
        for product in products:
            for region in regions:
                sales = np.random.randint(100, 1000)
                data.append([date, product, region, sales])

    df = pd.DataFrame(data, columns=['Date', 'Product', 'Region', 'Sales'])
    df.to_csv('data/sample_data.csv', index=False)
    print("Sample data generated successfully!")

if __name__ == '__main__':
    generate_sample_data()