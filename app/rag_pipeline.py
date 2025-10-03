import json, faiss, numpy as np, os

INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/metadata.json"

def build_or_load_index(embeddings):
    """Build or load FAISS index + metadata"""
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        return index, metadata

    if not embeddings:
        raise ValueError("No embeddings provided and no saved index found.")

    # Build new index
    dimension = len(embeddings[0][0])
    index = faiss.IndexFlatL2(dimension)
    vecs = np.array([vec for vec, _, _ in embeddings]).astype("float32")
    index.add(vecs)

    # Save index + metadata
    faiss.write_index(index, INDEX_PATH)
    metadata = [{"chunk": chunk, "meta": meta} for _, chunk, meta in embeddings]
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    return index, metadata


def retrieve_chunks(query, bedrock_client, index, metadata, k=3):
    """Embed query with Titan and retrieve top-k chunks"""
    body = {"inputText": query}
    resp = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps(body)
    )
    q_vec = np.array([json.loads(resp["body"].read())["embedding"]]).astype("float32")

    distances, indices = index.search(q_vec, k)
    results = [metadata[i] for i in indices[0]]
    return results


def query_nova(query, retrieved_chunks, bedrock_client):
    """
    Query Nova Pro with retrieved context and request inline citations.
    retrieved_chunks: list of dicts with {"chunk": text, "meta": {"paper": str, "page": int}}
    """
    context = ""
    citation_map = {}
    for i, r in enumerate(retrieved_chunks, 1):
        citation_id = f"[{i}]"
        context += f"{citation_id} {r['chunk']}\n(Source: {r['meta']['paper']}, page {r['meta']['page']})\n\n"
        citation_map[citation_id] = r["meta"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        f"Use the context below to answer the user’s question.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        f"Instructions: Provide a clear answer. "
                        f"Include inline citations like [1], [2] that match the sources. "
                        f"Do not invent references — only cite from provided sources."
                    )
                }
            ]
        }
    ]

    body = {
        "messages": messages,
        "inferenceConfig": {"maxTokens": 500, "temperature": 0.3}
    }

    resp = bedrock_client.invoke_model(
        modelId="amazon.nova-pro-v1:0",
        body=json.dumps(body)
    )
    answer_text = json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"]

    return answer_text, citation_map
