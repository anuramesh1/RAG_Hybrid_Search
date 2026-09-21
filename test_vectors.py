import chromadb

client = chromadb.PersistentClient(path="chroma_data")
collection = client.get_or_create_collection("rag_docs")

result = collection.get(limit=1, offset=3, include=["metadatas", "documents", "embeddings"])

print('ID:', result['ids'][0]);
print('Metadata:', result['metadatas'][0]);
print('Document:', result['documents'][0]);
print('Embedding:', result['embeddings'][0]);