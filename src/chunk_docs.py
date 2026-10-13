from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "data/documents/ed3book.pdf"

# Create PDF loader
loader = PyPDFLoader(pdf_path)

# Load the PDF
documents = loader.load()

print("Number of pages:", len(documents))

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nFirst chunk metadata:")
print(chunks[0].metadata)