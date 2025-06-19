# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:40:39 2025

@author: rahul
"""

# deploy_model.py
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import get_execution_role
import boto3

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()  # IAM role with SageMaker permissions
bucket_name = 'your-weather-bucket'
region = boto3.Session().region_name

# Define model parameters
model_artifact = f's3://{bucket_name}/models/model.tar.gz'
image_uri = sagemaker.image_uris.retrieve(
    framework='sklearn',
    region=region',
    version='1.2-1'
)

# Create SageMaker model
sklearn_model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point='inference.py',
    source_dir='code',
    framework_version='1.2.1',
    sagemaker_session=sagemaker_session
)

print(f"Model created with artifact: {model_artifact}")

# Continue in deploy_model.py
# Deploy to endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t3.medium',
    endpoint_name='weather-sales-forecast-endpoint'
)

# Configure predictor to handle JSON input/output
predictor.accept = 'application/json'
predictor.content_type = 'application/json'

print(f"Endpoint deployed: {predictor.endpoint_name}")

#predictor.delete_model()
#predictor.delete_endpoint()