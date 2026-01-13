"""
Sample Data Generator for Credit Card Fraud Detection
Creates realistic test data matching the Kaggle dataset structure
Use this for testing if the real dataset is not yet downloaded
"""

import csv
import random
import os

def generate_sample_data(num_records=10000, fraud_ratio=0.002):
    """Generate sample credit card fraud data"""
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'creditcard.csv')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Column names matching Kaggle dataset
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    
    print(f"Generating {num_records} sample transactions...")
    print(f"Fraud ratio: {fraud_ratio*100:.2f}%")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        fraud_count = 0
        normal_count = 0
        
        for i in range(num_records):
            # Time: seconds from start (up to 2 days)
            time_val = random.uniform(0, 172800)
            
            # Determine if fraud
            is_fraud = random.random() < fraud_ratio
            
            # V1-V28: PCA components (different distributions for fraud vs normal)
            if is_fraud:
                # Fraud transactions have different patterns
                v_features = [random.gauss(-2, 3) for _ in range(28)]
                amount = random.uniform(1, 2000)  # Often unusual amounts
                fraud_count += 1
            else:
                # Normal transactions
                v_features = [random.gauss(0, 1) for _ in range(28)]
                amount = random.uniform(1, 500)  # Typical amounts
                normal_count += 1
            
            # Build row
            row = [time_val] + v_features + [round(amount, 2), 1 if is_fraud else 0]
            writer.writerow(row)
            
            if (i + 1) % 2000 == 0:
                print(f"  Generated {i + 1}/{num_records} records...")
    
    print(f"\n✓ Sample data generated successfully!")
    print(f"  Location: {output_path}")
    print(f"  Normal transactions: {normal_count}")
    print(f"  Fraud transactions: {fraud_count}")
    print(f"  Total: {num_records}")
    
    return output_path

if __name__ == "__main__":
    generate_sample_data(num_records=50000, fraud_ratio=0.002)
