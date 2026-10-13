from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example text
text = "Self-attention allows a model to understand relationships between words."

# Convert text into an embedding
embedding = model.encode(text)

# Display the result
print("Embedding:")
print(embedding)

print("\nNumber of dimensions:")
print(len(embedding))