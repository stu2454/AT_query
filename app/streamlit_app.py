import streamlit as st
from embeddings import embed_multiple_papers
from rag_pipeline import build_or_load_index, retrieve_chunks, query_nova
import boto3, os, base64
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# -------------------------------
# AWS credentials handling
# -------------------------------
if os.path.exists(".env"):
    # Local dev
    load_dotenv()
    aws_access = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
else:
    # Streamlit Cloud
    aws_access = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
    aws_region = st.secrets.get("AWS_DEFAULT_REGION", "ap-southeast-2")

runtime = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access,
    aws_secret_access_key=aws_secret
)

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="Assistive Technology Knowledge Finder", layout="wide")
st.title("🦾 Assistive Technology Knowledge Finder (Titan + Nova Pro)")

uploaded_files = st.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)
query = st.text_input("Ask a question about your papers")

# Ensure data directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("papers", exist_ok=True)

# -------------------------------
# File management
# -------------------------------
file_paths = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("papers", uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

# -------------------------------
# Load or build FAISS index
# -------------------------------
if os.path.exists("data/faiss_index.bin"):
    index, metadata = build_or_load_index([])  # Load existing
    index_status = "✅ Loaded existing index"
else:
    if file_paths:
        with st.spinner("Building FAISS index for uploaded PDFs..."):
            embeddings = embed_multiple_papers(file_paths, runtime)
            index, metadata = build_or_load_index(embeddings)
        index_status = "✅ New index built"
    else:
        index, metadata = None, None
        index_status = "⚠️ No index available"

# -------------------------------
# Sidebar Knowledge Base
# -------------------------------
st.sidebar.header("📂 Knowledge Base")
if file_paths:
    for file_path in file_paths:
        try:
            reader = PdfReader(file_path)
            page_count = len(reader.pages)
        except Exception:
            page_count = "?"
        st.sidebar.markdown(
            f"- **{os.path.basename(file_path)}** ({page_count} pages) – "
            f"{'Indexed ✅' if index else 'Pending'}"
        )
else:
    st.sidebar.info("No PDFs uploaded yet.")

st.sidebar.markdown("---")
st.sidebar.text(index_status)

if st.sidebar.button("🗑️ Clear Index"):
    if os.path.exists("data/faiss_index.bin"):
        os.remove("data/faiss_index.bin")
    if os.path.exists("data/metadata.json"):
        os.remove("data/metadata.json")
    st.sidebar.warning("Index cleared. Re-upload or rebuild required.")

# -------------------------------
# Rebuild index button
# -------------------------------
if st.button("🔄 Rebuild Index") and file_paths:
    with st.spinner("Rebuilding FAISS index..."):
        embeddings = embed_multiple_papers(file_paths, runtime)
        index, metadata = build_or_load_index(embeddings)
    st.success("✅ Index rebuilt successfully")

# -------------------------------
# Utility to embed PDFs inline
# -------------------------------
def display_pdf(file_path, height=400):
    """Embed a PDF in Streamlit using base64 encoding"""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="{height}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# -------------------------------
# Handle user query
# -------------------------------
if query and index and metadata:
    with st.spinner("Searching and querying Nova Pro..."):
        retrieved = retrieve_chunks(query, runtime, index, metadata, k=3)
        answer, citation_map = query_nova(query, retrieved, runtime)

        st.markdown("### 💡 Answer")
        st.write(answer)

        st.markdown("### 🔗 Citations")
        for cid, meta in citation_map.items():
            paper_path = os.path.join("papers", meta["paper"])
            if os.path.exists(paper_path):
                st.markdown(f"**{cid} – {meta['paper']} (page {meta['page']})**")
                display_pdf(paper_path, height=300)
            else:
                st.markdown(f"{cid} {meta['paper']} (page {meta['page']})")

        st.markdown("---")
        st.markdown("### 📚 Retrieved Context")
        for r in retrieved:
            st.text_area(
                f"{r['meta']['paper']} (page {r['meta']['page']})",
                r["chunk"],
                height=100,
            )
