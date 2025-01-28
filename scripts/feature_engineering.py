# scripts/feature_engineering.py

import pandas as pd
import numpy as np
import os

def load_data(file_path="./data/processed/cleaned_data.csv"):
    """
    Loads cleaned data and ensures required columns are present.
    """
    df = pd.read_csv(file_path)
    return df

def create_proxy_default_label(df):
    """
    Creates a proxy default label (0 = good, 1 = bad) based on RFM scoring or another logic.
    Modify this logic according to your dataset.
    """
    # Example: Flag as "bad" if total spending (Amount_sum_x) is below median.
    df["proxy_default_label"] = df["Amount_sum_x"].apply(lambda x: 1 if x < df["Amount_sum_x"].median() else 0)

    print("\n✅ Proxy Default Label Created:")
    print(df["proxy_default_label"].value_counts())  # Show label distribution
    return df

def save_data(df, save_path="./data/processed/cleaned_data.csv"):
    """
    Saves the updated DataFrame with the proxy label.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n✅ Data saved to {save_path}")

def main():
    # 1. Load existing processed data
    df = load_data()

    # 2. Create proxy default label
    df = create_proxy_default_label(df)

    # 3. Save updated data
    save_data(df)

if __name__ == "__main__":
    main()
