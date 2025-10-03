from PyPDF2 import PdfReader
import json, os

def embed_paper(file_path, bedrock_client):
    """Embed a single PDF into (vector, chunk, metadata) tuples"""
    reader = PdfReader(file_path)
    all_chunks = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        # Simple chunking by 1000 chars
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            body = {"inputText": chunk}
            resp = bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps(body)
            )
            vec = json.loads(resp["body"].read())["embedding"]
            all_chunks.append((vec, chunk, {
                "paper": os.path.basename(file_path),
                "page": page_num + 1
            }))
    return all_chunks

def embed_multiple_papers(file_paths, bedrock_client):
    """Embed multiple PDFs into a combined list"""
    all_embeddings = []
    for path in file_paths:
        all_embeddings.extend(embed_paper(path, bedrock_client))
    return all_embeddings
