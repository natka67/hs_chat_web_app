from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import io
import zipfile
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

load_dotenv()

# --- Azure Storage Settings ---
storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
storage_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

connect_str = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(storage_container_name)

router = APIRouter()

@router.get("/download-pdfs")
async def download_all_pdfs(folder: str = Query("input_pdf/")):
    blob_list = container_client.list_blobs(name_starts_with=folder)
    pdf_blobs = [blob.name for blob in blob_list if blob.name.endswith(".pdf")]

    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, mode="w") as zip_file:
        for blob_name in pdf_blobs:
            blob_client = container_client.get_blob_client(blob_name)
            file_stream = io.BytesIO()
            download_stream = blob_client.download_blob()
            file_stream.write(download_stream.readall())
            file_stream.seek(0)

            filename = blob_name.split("/")[-1]
            zip_file.writestr(filename, file_stream.read())

    zip_stream.seek(0)

    response = StreamingResponse(zip_stream, media_type="application/x-zip-compressed")
    response.headers["Content-Disposition"] = "attachment; filename=all_pdfs.zip"
    return response