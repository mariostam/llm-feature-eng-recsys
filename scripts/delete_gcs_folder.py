
from google.cloud import storage

def delete_gcs_folder():
    """
    Deletes a folder and its contents from a GCS bucket.
    """
    bucket_name = 'llm-feature-engineering-thesis-bucket'
    folder_name = 'test_batches'
    project_id = 'llm-feature-engineering-thesis'

    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=folder_name))

        if not blobs:
            print(f"Folder gs://{bucket_name}/{folder_name}/ does not exist or is already empty.")
            return

        print(f"Deleting {len(blobs)} files from gs://{bucket_name}/{folder_name}/...")
        for blob in blobs:
            blob.delete()
        
        print("Deletion complete.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    delete_gcs_folder()
