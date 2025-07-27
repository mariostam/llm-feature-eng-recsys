

import os
from google.cloud import storage

def upload_dummy_batches():
    """
    Uploads the local dummy batch files to Google Cloud Storage.
    """
    local_dir = 'data/dummy_batches'
    bucket_name = 'llm-feature-engineering-thesis-bucket'
    gcs_dir = 'test_batches'

    try:
        storage_client = storage.Client(project='llm-feature-engineering-thesis')
        bucket = storage_client.bucket(bucket_name)

        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            if os.path.isfile(local_path):
                blob_name = f"{gcs_dir}/{filename}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")

        print("\nUpload complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have authenticated with 'gcloud auth application-default login'.")

if __name__ == '__main__':
    upload_dummy_batches()

