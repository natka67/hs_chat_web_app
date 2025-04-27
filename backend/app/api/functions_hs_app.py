
import os
import PyPDF2
import tiktoken
import numpy as np
from tqdm import tqdm  # NEW
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
from openai import RateLimitError
from azure.storage.blob import BlobServiceClient
import io
import re
from azure.core.exceptions import ResourceNotFoundError
import pandas as pd
# --- Load environment variables ---
load_dotenv()

# --- OpenAI Chat settings ---
openai_endpoint = os.getenv("ENDPOINT_URL")
openai_deployment_model = os.getenv("DEPLOYMENT_NAME")
openai_key = os.getenv("MODEL_KEY")

# --- OpenAI Embedding settings ---
embed_endpoint = os.getenv("ENDPOINT_EMBED")
embed_deployment_model = os.getenv("DEPLOYMENT_EMBED_NAME")
embed_key = os.getenv("MODEL_KEY_EMBED")

# --- Azure Storage Settings ---
storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
storage_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


connect_str = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(storage_container_name)

# --- Initialize Azure OpenAI clients ---
print("🔵 Initializing Azure OpenAI clients...")
def safe_azure_openai(azure_endpoint: str, api_key: str, api_version: str = "2025-01-01-preview") -> AzureOpenAI:
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(proxy_var, None)
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )

print("🔵 Initializing Azure OpenAI clients...")

client_chat = safe_azure_openai(
    azure_endpoint=openai_endpoint,
    api_key=openai_key,
)

client_embed = safe_azure_openai(
    azure_endpoint=embed_endpoint,
    api_key=embed_key,
)

# --- 1. Read and extract text from PDFs ---
def extract_text_from_pdfs_from_azure(folder_prefix="input_pdf/"):
    print(f"📄 Loading PDFs from Azure folder: {folder_prefix}")
    all_text = ""

    blob_list = container_client.list_blobs(name_starts_with=folder_prefix)
    pdf_blobs = [blob.name for blob in blob_list if blob.name.endswith(".pdf")]

    print(f"🔹 Found {len(pdf_blobs)} PDF files in Azure.")

    for blob_name in tqdm(pdf_blobs, desc="📄 Extracting PDFs from Azure"):
        blob_client = container_client.get_blob_client(blob_name)
        pdf_stream = io.BytesIO()
        download_stream = blob_client.download_blob()
        pdf_stream.write(download_stream.readall())
        pdf_stream.seek(0)

        reader = PyPDF2.PdfReader(pdf_stream)
        for page in reader.pages:
            all_text += page.extract_text() + "\n"

    return all_text


# --- 2. Chunk text intelligently ---
def chunk_text(text, max_tokens=500):
    print("✂️  Splitting extracted text into chunks...")
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in tqdm(words, desc="✂️ Chunking words"):
        current_chunk.append(word)
        current_length = len(tokenizer.encode(" ".join(current_chunk)))
        if current_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"🔹 Created {len(chunks)} text chunks.")
    return chunks

# --- 3. Embed chunks ---
def embed_texts(texts, batch_size=5):
    print("🧠 Embedding text chunks...")
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="🧠 Embedding batches"):
        batch = texts[i:i+batch_size]
        
        success = False
        while not success:
            try:
                response = client_embed.embeddings.create(
                    model=embed_deployment_model,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]
                embeddings.extend(batch_embeddings)
                success = True
                time.sleep(1.5)  # Small delay between batches to avoid hitting rate limit
            except RateLimitError as e:
                print("⚠️ Rate limit hit. Sleeping for 60 seconds...")
                time.sleep(60)  # Wait longer and retry

    print("✅ Embedding completed.")
    return embeddings

# --- 4. Search for the most relevant chunk ---
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_similar_chunks(question, chunks, chunk_embeddings):
    print("🔍 Searching for most relevant text chunk...")
    question_embed = client_embed.embeddings.create(
        model=embed_deployment_model,
        input=[question]
    ).data[0].embedding

    similarities = [cosine_similarity(question_embed, emb) for emb in chunk_embeddings]
    top_idx = np.argmax(similarities)
    print(f"✅ Found the most relevant chunk (index {top_idx}).")
    return chunks[top_idx]

def save_embeddings_to_azure(chunks, embeddings, blob_name="embeddings/embeddings.npz"):
    print(f"📤 Saving embeddings to Azure as '{blob_name}'...")
    buffer = io.BytesIO()
    np.savez(buffer, chunks=chunks, embeddings=np.array(embeddings))
    buffer.seek(0)

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(buffer, overwrite=True)

    print(f"✅ Embeddings uploaded to Azure Blob Storage at '{blob_name}'.")


def load_embeddings_from_azure(blob_name="embeddings/embeddings.npz"):
    print(f"📥 Loading embeddings from Azure blob '{blob_name}'...")
    blob_client = container_client.get_blob_client(blob_name)

    try:
        stream = io.BytesIO()
        download_stream = blob_client.download_blob()
        stream.write(download_stream.readall())
        stream.seek(0)

        data = np.load(stream, allow_pickle=True)
        chunks = data["chunks"].tolist()
        embeddings = data["embeddings"]

        print(f"✅ Embeddings loaded from Azure.")
        return chunks, embeddings
    except ResourceNotFoundError:
        print(f"❌ Embeddings blob '{blob_name}' not found.")
        raise



def extract_hs_data_from_text(text):
    entries = []

    # Capture the sections and chapters
    current_section = None
    current_chapter = None

    section_pattern = re.compile(r'SECTION\s+[IVXLCDM]+\s+(.+?)\n', re.MULTILINE)
    chapter_pattern = re.compile(r'Chapter\s+\d+\s*(.+?)\n', re.MULTILINE)
    row_pattern = re.compile(r'(\d{2}\.\d{2}(?:\.\d{2})?)\s+(.+?)\s+(\d+)\s*$', re.MULTILINE)

    lines = text.splitlines()
    print(f"📄 Total lines to process: {len(lines)}")

    for line in tqdm(lines, desc="🔍 Processing lines"):
        line = line.strip()

        if section_match := section_pattern.search(line):
            current_section = section_match.group(1).strip()
            print(f"📚 Found new section: {current_section}")

        if chapter_match := chapter_pattern.search(line):
            current_chapter = chapter_match.group(1).strip()
            print(f"📖 Found new chapter: {current_chapter}")

        if row_match := row_pattern.match(line):
            hs_code = row_match.group(1)
            description = row_match.group(2)
            duty = int(row_match.group(3))

            formatted_entry = {
                "hs_code": hs_code,
                "description": description,
                "duty": duty,
                "section": current_section,
                "chapter": current_chapter,
                "explanatory_notes": None,
                "statistical_suffix": None
            }

            print("🔹 Line matched:")
            print(f"    Raw Line     : {line}")
            print(f"    Parsed Entry : {formatted_entry}")

            entries.append(formatted_entry)

    print(f"✅ Total entries extracted: {len(entries)}")
    df = pd.DataFrame(entries)
    return df

import json

def send_query_to_chat(question, relevant_chunk):
    # --- 1. Build Chat Prompt ---
    chat_prompt = [
        {
            "role": "system",
            "content": (
                "You are a customs classification assistant. Given a product description and relevant context, determine the most appropriate Harmonized System (HS) code. Always provide at least the first 6 digits of the HS code. If uncertain about the full code beyond 6 digits, list all plausible 8-digit codes in the 'notes' field. Respond strictly in JSON format with the following fields: 'hs_code' (string), 'confidence' (float between 0 and 1), and 'notes' (string). In the 'notes' field, include a detailed explanation of the classification decision, referencing relevant chapters, headings, subheadings and explanatory notes. Discuss any applicable General Rules of Interpretation (GRIs) and provide reasoning for the selected code, especially in cases where multiple codes could apply."
                "Respond with a JSON object containing the following fields: "
                "`hs_code` (string), `confidence` (float between 0 and 1), and `notes` (string)."
            )
        },
        {
            "role": "system",
            "content": f"Context: {relevant_chunk}"
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # --- 2. Call Chat Completion ---
    print("💬 Sending request to Chat Completion...")
    completion = client_chat.chat.completions.create(
        model=openai_deployment_model,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.3,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        response_format={"type": "json_object"}  # Ensure the response is in JSON format
    )

    # --- 3. Parse and Return JSON Response ---
    response_content = completion.choices[0].message.content
    try:
        response_json = json.loads(response_content)
        return response_json
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        return {
            "hs_code": "Unknown",
            "confidence": 0.0,
            "notes": "Failed to parse model response."
        }
