import boto3
import botocore
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--access_key', help='AWS access key')
parser.add_argument('-s', '--secret_key', help='AWS secret key')
parser.add_argument('-t', '--session_token', help='AWS session token')
parser.add_argument('-b', '--bucket', help='S3 bucket name', required=True)
parser.add_argument('-k', '--key', help='S3 key', required=True)
parser.add_argument('-f', '--file', help='file to upload', required=True)


def main():
    args = parser.parse_args()


    credentials = {}
    if args.access_key is not None:
        credentials['aws_access_key_id'] = args.access_key
    if args.secret_key is not None:
        credentials['aws_secret_access_key'] = args.secret_key
    if args.session_token is not None:
        credentials['aws_session_token'] = args.session_token

    session = boto3.Session(**credentials)
    s3 = session.resource('s3')
    try:
        s3.Bucket(args.bucket).download_file(args.key, args.file)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

if __name__=='__main__':
    main()