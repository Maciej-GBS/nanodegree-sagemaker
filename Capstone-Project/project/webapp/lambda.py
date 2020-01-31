import json
import boto3

def lambda_handler(event, context):
    runtime = boto3.Session().client('sagemaker-runtime')

    response = runtime.invoke_endpoint(EndpointName = '**ENDPOINT NAME**',
                                       ContentType = 'text/plain',
                                       Body = event['body'])

    result = json.loads(response['Body'].read())

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'application/json', 'Access-Control-Allow-Origin' : '*' },
        'body' : result
    }
