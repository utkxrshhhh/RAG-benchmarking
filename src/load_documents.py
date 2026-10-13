from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/documents/ed3book.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))

print("\nFirst page content:")
print(documents[0].page_content)

print("\nMetadata:")
print(documents[0].metadata)