import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load API key from .env
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

# Connect to Pinecone
pc = Pinecone(api_key=api_key)

# Connect to our existing index
index = pc.Index("rag-evaluation-1000-100")

# Load PDF
pdf_path = "data/documents/ed3book.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Pages:", len(documents))


# Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract text
texts = [chunk.page_content for chunk in chunks]

# Create embeddings
embeddings = model.encode(texts)

print("Embeddings created:", len(embeddings))


# Prepare vectors
vectors = []

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

    vector = {
        "id": f"chunk_{i}",
        "values": embedding.tolist(),
        "metadata": {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page")
        }
    }

    vectors.append(vector)


# Upload in batches
batch_size = 100

for start in range(0, len(vectors), batch_size):

    batch = vectors[start:start + batch_size]

    index.upsert(vectors=batch)

    print(
        f"Uploaded {min(start + batch_size, len(vectors))}"
        f"/{len(vectors)}"
    )


print("All vectors uploaded successfully!")