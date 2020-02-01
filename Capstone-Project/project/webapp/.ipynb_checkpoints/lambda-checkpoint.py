import json
import boto3
import os

def lambda_handler(event, context):
    runtime = boto3.Session().client('sagemaker-runtime')

    endpoint = os.environ['PREDICTOR']
    response = runtime.invoke_endpoint(EndpointName = "**ENDPOINT**",
                                       ContentType = 'text/plain',
                                       Body = event['body'])

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'application/json', 'Access-Control-Allow-Origin' : '*' },
        'body' : response['Body']
    }

