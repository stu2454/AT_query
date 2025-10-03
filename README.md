# 🦾 Assistive Technology Knowledge Finder (Titan + Nova Pro via AWS Bedrock)

This project is a prototype **knowledge retrieval assistant** designed to explore and interrogate Assistive Technology (AT) knowledge sources (e.g., research papers, policy documents).

It uses:
- **Titan Text Embeddings v2** → convert documents into vector embeddings.
- **Nova Pro** → answer questions using retrieval-augmented generation (RAG).
- **Streamlit** → interactive UI.
- **Docker** → containerised deployment.

---

## 🚀 How to Run

### 1. Local (Docker)
```bash
docker-compose up --build
```
Then open http://localhost:8501

### 2. Deployment Options
- **Streamlit Cloud** (simplest): Push repo to GitHub and connect.
- **Render**: Deploy using the Dockerfile.
- **AWS Amplify / ECS**: Suitable for enterprise or scaling.

---

## 🛠 Features
- Upload **multiple PDFs** at once (e.g. research papers, AT standards).
- Automatic **chunking + embedding** of text using Titan.
- Semantic **retrieval across all papers** with FAISS vector search.
- Nova Pro generates answers grounded in retrieved context.
- Shows which **document + page** the answer came from.
- **Inline citations** in answers (e.g., [1], [2]) linked to embedded PDF previews.
- **Sidebar knowledge base** lists uploaded PDFs with page counts and indexing status (Indexed ✅ / Pending).
- **Rebuild index button** to regenerate embeddings when adding new documents.
- **Clear index button** in sidebar to reset the knowledge base.
- **Persistent local storage** of FAISS index and metadata (documents remain indexed between sessions).

---

## 📂 Project Structure
```
app/
  ├─ streamlit_app.py      # Main Streamlit UI (with sidebar + PDF preview)
  ├─ embeddings.py         # Titan embeddings + PDF chunking
  ├─ rag_pipeline.py       # Retrieval + Nova Pro query (citations supported)
  └─ utils.py              # Utilities / placeholders

papers/                    # Upload PDFs here (mounted into container)
data/                      # FAISS index + metadata persistence
Dockerfile
docker-compose.yml
requirements.txt
.env                       # AWS credentials + region (local only)
```

---

## 🔐 Environment Variables
Set your AWS Bedrock credentials in `.env` (local dev) or in **Streamlit Cloud Secrets** (deployment):

```env
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=ap-southeast-2
```

---

## ⚠️ Notes
- Uses **on-demand models** only (Titan + Nova Pro).  
- Claude 3.7 Sonnet requires an **inference profile** and is not included here.  
- For real deployments, consider adding **persistent cloud storage** (e.g., S3 + DynamoDB) for indexed documents.
