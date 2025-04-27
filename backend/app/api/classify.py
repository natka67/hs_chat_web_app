from fastapi import APIRouter
from app.models.schemas import ClassifyRequest
from app.api.functions_hs_app import (
    load_embeddings_from_azure,
    extract_text_from_pdfs_from_azure,
    chunk_text,
    embed_texts,
    save_embeddings_to_azure,
    search_similar_chunks,
    send_query_to_chat,
)

router = APIRouter()

# 📂 Phase 1: Load embeddings at server start
print("📂 Setting up embeddings...")

try:
    text_chunks, chunk_embeddings = load_embeddings_from_azure()
except Exception as e_azure_load:
    print(f"⚠️ Could not load embeddings from Azure: {e_azure_load}")
    try:
        print("📂 No embeddings found. Trying to load input PDFs from Azure...")
        all_text = extract_text_from_pdfs_from_azure(folder_prefix="pdfs/")
        text_chunks = chunk_text(all_text)
        chunk_embeddings = embed_texts(text_chunks)
        save_embeddings_to_azure(text_chunks, chunk_embeddings)
    except Exception as e_pdf_azure:
        print(f"❌ Critical error: Could not load input PDFs from Azure: {e_pdf_azure}")
        raise RuntimeError("Failed to load input PDFs and generate embeddings from Azure.") from e_pdf_azure

print("✅ Embeddings ready!")

@router.post("/classify")
async def classify(request: ClassifyRequest):
    description = request.description
    print(f"🔍 Classifying description: {description}")

    # Search relevant chunk
    relevant_chunk = search_similar_chunks(description, text_chunks, chunk_embeddings)

    # Get answer
    answer = send_query_to_chat(description, relevant_chunk)

    return answer
