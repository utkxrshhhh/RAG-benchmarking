from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Path to the PDF
pdf_path = "data/documents/ed3book.pdf"

# Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Number of pages:", len(documents))

# Create the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split the PDF into chunks
chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get the text from every chunk
texts = [chunk.page_content for chunk in chunks]

# Convert all chunks into embeddings
embeddings = model.encode(texts)

print("Embeddings created successfully!")

print("Number of embeddings:", len(embeddings))
print("Dimensions of each embedding:", len(embeddings[0]))