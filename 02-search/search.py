from langchain_community.document_loaders import PyPDFLoader

file_path = "/Users/jason/Github/hello-langchain/02-search/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

import getpass
import os

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3.2-vision",base_url="http://192.168.99.41:11434")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

print(ids)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])

# results = await vector_store.asimilarity_search("When was Nike incorporated?")

# print(results[0])