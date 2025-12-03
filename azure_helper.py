import ssl
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO
import joblib
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_csv_from_blob(connection_string: str, container_name: str, blob_name: str) -> pd.DataFrame:
    try:
        # Secure SSL context

        # Create client with SSL context
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Get blob client
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download blob content
        blob_data = blob_client.download_blob()
        csv_content =  blob_data.readall().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        logger.info("csv loaded")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV from blob: {e}")
        return pd.DataFrame()



def read_model_from_blob(connection_string: str, container_name: str, blob_name: str) -> pd.DataFrame:
    try:
        # Secure SSL context

        # Create client with SSL context
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Get blob client
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download blob content
        blob_data = blob_client.download_blob()
        data =  blob_data.readall()
        logger.info("mode file loaded")
        return data
    except Exception as e:
        logger.error(f"Error reading model file from blob: {e}")
        return None
    

def read_json_from_blob(connection_string, container_name, blob_name):
    """
    Reads a JSON file from Azure Blob Storage and returns it as a Python object (dict or list).
    """

    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)

    # Get the blob client
    blob_client = container_client.get_blob_client(blob_name)

    # Download the blob content
    blob_data = blob_client.download_blob().readall()

    # Decode bytes to string and load JSON
    json_data = json.loads(blob_data.decode("utf-8"))
    logger.info("json file loaded")

    return json_data