import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

def load_data():
    """
    Load the cleaned dataset from the processed folder.
    """
    data_path = "/Users/mulsewsmba/Downloads/WK-6 Credit Scoring /data/processed/cleaned_data.csv"
    data = pd.read_csv(data_path)
    return data

def create_aggregate_features(data):
    """
    Create aggregate features for each customer.
    """
    # Group by CustomerId to create aggregate features
    aggregate_features = data.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],  # Total, average, std, and count of transactions
        'Value': ['sum', 'mean', 'std']  # Total, average, and std of transaction values
    })

    # Flatten the multi-level column index
    aggregate_features.columns = ['_'.join(col).strip() for col in aggregate_features.columns.values]

    # Reset index to merge with the original data
    aggregate_features.reset_index(inplace=True)

    return aggregate_features

def extract_temporal_features(data):
    """
    Extract temporal features from the TransactionStartTime column.
    """
    # Convert TransactionStartTime to datetime
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    # Extract hour, day, month, and year
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year

    return data

def encode_categorical_variables(data):
    """
    Encode categorical variables using One-Hot Encoding or Label Encoding.
    """
    # One-Hot Encoding for ProductCategory
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    product_category_encoded = one_hot_encoder.fit_transform(data[['ProductCategory']])
    product_category_encoded_df = pd.DataFrame(
        product_category_encoded,
        columns=one_hot_encoder.get_feature_names_out(['ProductCategory'])
    )

    # Label Encoding for ChannelId
    label_encoder = LabelEncoder()
    data['ChannelId_encoded'] = label_encoder.fit_transform(data['ChannelId'])

    # Concatenate encoded features with the original data
    data = pd.concat([data, product_category_encoded_df], axis=1)

    return data

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    """
    # Fill missing numerical values with the median
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_features:
        data[col] = data[col].fillna(data[col].median())  # Updated to avoid chained assignment

    # Fill missing categorical values with the mode
    categorical_features = data.select_dtypes(include=['object']).columns
    for col in categorical_features:
        data[col] = data[col].fillna(data[col].mode()[0])  # Updated to avoid chained assignment

    return data
def normalize_numerical_features(data):
    """
    Normalize/Standardize numerical features.
    """
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

    # Standardize numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

def save_cleaned_data(data):
    """
    Save the processed dataset to the processed folder.
    """
    cleaned_data_path = "/Users/mulsewsmba/Downloads/WK-6 Credit Scoring /data/processed/cleaned_data.csv"
    data.to_csv(cleaned_data_path, index=False)
    print(f"Processed data saved to {cleaned_data_path}")

def main():
    """
    Main function to execute feature engineering tasks.
    """
    # Load the cleaned data
    data = load_data()

    # Create aggregate features
    aggregate_features = create_aggregate_features(data)

    # Merge aggregate features with the original data
    data = data.merge(aggregate_features, on='CustomerId', how='left')

    # Extract temporal features
    data = extract_temporal_features(data)

    # Encode categorical variables
    data = encode_categorical_variables(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Normalize numerical features
    data = normalize_numerical_features(data)

    # Save the processed data
    save_cleaned_data(data)

if __name__ == "__main__":
    main()