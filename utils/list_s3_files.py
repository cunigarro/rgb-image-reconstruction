import boto3

def list_s3_files(bucket_name, prefix):
    s3 = boto3.client('s3')
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.jpg'):
                    keys.append(obj['Key'])
    return sorted(keys)
