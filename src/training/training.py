import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import boto3
import pickle

# Load datasets

expanded_data_path = os.environ.get('EXPANDED_DATA_PATH', '/usr/src/app/data/Expanded_data_with_more_features.csv')
original_data_path = os.environ.get('ORIGINAL_DATA_PATH', '/usr/src/app/data/Original_data_with_more_rows.csv')
bucket_name = os.environ.get('BUCKET_NAME', '')
origin_model_file_name = os.environ.get('origin_model_file_name', '')

expanded_data = pd.read_csv(expanded_data_path)
original_data = pd.read_csv(original_data_path)

# Data cleaning and preprocessing for expanded data
expanded_data['EthnicGroup'].fillna(expanded_data['EthnicGroup'].mode()[0], inplace=True)
expanded_data['ParentEduc'].fillna(expanded_data['ParentEduc'].mode()[0], inplace=True)
expanded_data['TestPrep'].fillna(expanded_data['TestPrep'].mode()[0], inplace=True)
expanded_data['ParentMaritalStatus'].fillna(expanded_data['ParentMaritalStatus'].mode()[0], inplace=True)
expanded_data['PracticeSport'].fillna(expanded_data['PracticeSport'].mode()[0], inplace=True)
expanded_data['IsFirstChild'].fillna(expanded_data['IsFirstChild'].mode()[0], inplace=True)
expanded_data['TransportMeans'].fillna(expanded_data['TransportMeans'].mode()[0], inplace=True)
expanded_data['WklyStudyHours'].fillna(expanded_data['WklyStudyHours'].mode()[0], inplace=True)
expanded_data['NrSiblings'].fillna(expanded_data['NrSiblings'].mean(), inplace=True)

expanded_data.drop(columns=['Unnamed: 0'], inplace=True)

expanded_data_encoded = pd.get_dummies(expanded_data, drop_first=True)

scaler = StandardScaler()
numerical_features = ['NrSiblings', 'MathScore', 'ReadingScore', 'WritingScore']
expanded_data_encoded[numerical_features] = scaler.fit_transform(expanded_data_encoded[numerical_features])

# Ensure there are no missing values in the target variables
expanded_data_encoded = expanded_data_encoded.dropna(subset=['MathScore', 'ReadingScore', 'WritingScore'])

# Split data into training and test sets
X = expanded_data_encoded.drop(columns=['MathScore', 'ReadingScore', 'WritingScore'])
y_math = expanded_data_encoded['MathScore']
y_reading = expanded_data_encoded['ReadingScore']
y_writing = expanded_data_encoded['WritingScore']

X_train, X_test, y_math_train, y_math_test = train_test_split(X, y_math, test_size=0.2, random_state=42)
_, _, y_reading_train, y_reading_test = train_test_split(X, y_reading, test_size=0.2, random_state=42)
_, _, y_writing_train, y_writing_test = train_test_split(X, y_writing, test_size=0.2, random_state=42)

# Train and evaluate models for MathScore
model_math = RandomForestRegressor(random_state=42)
model_math.fit(X_train, y_math_train)

# Save the model to a file
model_filename = origin_model_file_name + '.pkl'
backup_address = (
        origin_model_file_name + datetime.now().strftime("%Y%m%d%H%M%S") + '.pkl'
)
with open(model_filename, 'wb') as file:
    pickle.dump(model_math, file)

# Save the model to S3
dest_final_target = '/final-result/' + model_filename
dest_backup_target = '/backup-result/' + backup_address
s3 = boto3.client('s3')
s3.upload_file(model_filename, bucket_name, dest_final_target)
s3.upload_file(model_filename, bucket_name, dest_backup_target)

print("Model trained and saved to S3 successfully!")
