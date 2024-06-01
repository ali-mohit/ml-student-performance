import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import boto3
from io import BytesIO

app = Flask(__name__)


# Load the model from S3
def load_model_from_s3():
    bucket_name = os.environ.get('BUCKET_NAME', '')
    origin_model_file_name = os.environ.get('origin_model_file_name', '')
    model_filename = origin_model_file_name + '.pkl'
    s3_src_obj = '/final-result/' + model_filename

    s3 = boto3.client('s3')
    model_object = s3.get_object(Bucket=bucket_name, Key=s3_src_obj)
    return pickle.load(BytesIO(model_object['Body'].read()))


model = load_model_from_s3()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from POST request
    df = pd.DataFrame(data)  # Convert to DataFrame

    # Preprocessing similar to training
    df['EthnicGroup'].fillna(df['EthnicGroup'].mode()[0], inplace=True)
    df['ParentEduc'].fillna(df['ParentEduc'].mode()[0], inplace=True)
    df['TestPrep'].fillna(df['TestPrep'].mode()[0], inplace=True)
    df['ParentMaritalStatus'].fillna(df['ParentMaritalStatus'].mode()[0], inplace=True)
    df['PracticeSport'].fillna(df['PracticeSport'].mode()[0], inplace=True)
    df['IsFirstChild'].fillna(df['IsFirstChild'].mode()[0], inplace=True)
    df['TransportMeans'].fillna(df['TransportMeans'].mode()[0], inplace=True)
    df['WklyStudyHours'].fillna(df['WklyStudyHours'].mode()[0], inplace=True)
    df['NrSiblings'].fillna(df['NrSiblings'].mean(), inplace=True)

    df_encoded = pd.get_dummies(df, drop_first=True)
    scaler = StandardScaler()
    numerical_features = ['NrSiblings']
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

    prediction = model.predict(df_encoded)
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
