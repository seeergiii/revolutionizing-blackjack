from google.cloud import storage
import os


def save_model_to_gcloud(model_path: str, bucket_name: str) -> None:
    """
    Uploads a model file to Google Cloud Storage (GCS).

    Args:
        model_path (str): The local file path of the model to be uploaded.
        bucket_name (str): The name of the GCS bucket where the model will be stored.

    Returns:
        None: No return value.
    """
    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(model_path)

    print("✅ Model saved to Google Cloud Storage (GCS)")

    return None


def download_model_from_gcloud(model_path: str, bucket_name: str) -> None:
    """
    Downloads a model file from Google Cloud Storage (GCS).

    Args:
        model_path (str): The local file path where the model will be saved.
        bucket_name (str): The name of the GCS bucket from which the model will be downloaded.

    Returns:
        None: No return value.
    """
    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    blob.download_to_filename(model_path)

    print("✅ Model downloaded from Google Cloud Storage (GCS)")

    return None


def download_nested_gcloud_folder(bucket_name, prefix, local_folder) -> None:
    """
    Downloads files from a nested folder in Google Cloud Storage (GCS).

    Args:
        bucket_name (str): The name of the GCS bucket containing the files.
        prefix (str): The prefix (folder path) for the files to be downloaded.
        local_folder (str): The local directory where the files will be saved.

    Returns:
        None: No return value.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        destination_path = os.path.join(local_folder, blob.name.replace(prefix, ""))
        destination_folder = os.path.dirname(destination_path)

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        blob.download_to_filename(destination_path)

    print("✅ Training data downloaded from GCS")

    return None


def check_specific_model_in_bucket(model_name: str, bucket_name: str) -> bool:
    """
    Checks if a specific file exists in a Google Cloud Storage (GCS) bucket.

    Args:
        model_name (str): The name of the file to check for in the GCS bucket.
        bucket_name (str): The name of the GCS bucket to search in.

    Returns:
        bool: True if the file exists in the bucket, False otherwise.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)

    return blob.exists()
