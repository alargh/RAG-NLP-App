from langchain_community.embeddings import HuggingFaceHubEmbeddings
import numpy as np
from dotenv import load_dotenv
import os
from src.text_processing import preprocess_text

def load_keywords(file_path):
    with open(file_path, 'r') as f:
        keywords = f.read().splitlines()
    return keywords

def calculate_semantic_similarity(text1, text2, embeddings):
    embedding1 = embeddings.embed_query(text1)
    embedding2 = embeddings.embed_query(text2)

    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)


def get_embeddings_model():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    embeddings = HuggingFaceHubEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )
    return embeddings


def is_relevant_to_concert_tours(document, keywords, keyword_matches_threshold):
    preprocessed_sentences, original_sentences = preprocess_text(document)

    preprocessed_doc = ' '.join(preprocessed_sentences)

    keyword_matches = sum(1 for keyword in keywords if keyword in preprocessed_doc)

    if keyword_matches >= keyword_matches_threshold:
        return True

    embeddings = get_embeddings_model()

    domain_description = "This document is about upcoming concert tours, musical performances, venues, artists, bands, tickets, live music events, and music festivals in 2025-2026 "

    semantic_similarity = calculate_semantic_similarity(domain_description, document, embeddings)
    similarity_threshold = 0.5
    return semantic_similarity >= similarity_threshold


def analyze_concert_tour_document(document_text, keyword_matches_threshold):
    load_dotenv()

    file_path = os.path.join(os.path.dirname(__file__), '..', 'keywords_extraction', 'keywords.txt')

    keywords = load_keywords(file_path)
    return is_relevant_to_concert_tours(document_text, keywords, keyword_matches_threshold)