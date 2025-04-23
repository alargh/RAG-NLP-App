import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceHubEmbeddings

def get_vectors(chunks, existing_vectors=None):
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    embeddings = HuggingFaceHubEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        huggingfacehub_api_token=hf_token
    )

    if existing_vectors is None:
        vectors = FAISS.from_texts(texts=chunks, embedding=embeddings)
    else:
        existing_vectors.add_texts(chunks)
        vectors = existing_vectors

    return vectors