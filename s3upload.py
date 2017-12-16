import boto3
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

    s3 = boto3.client('s3', **credentials)

    s3.upload_file(args.file, args.bucket, args.key)


if __name__=='__main__':
    main()