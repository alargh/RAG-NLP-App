import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.text_processing import preprocess_text

def summarize(text, thresh_ratio=0.5):
    filtered, original = preprocess_text(text)

    if len(filtered) <= 3:
        return text

    vec = TfidfVectorizer()
    mat = vec.fit_transform(filtered)
    scores = np.array(mat.sum(axis=1)).flatten()

    max_score = scores.max()
    threshold = thresh_ratio * max_score

    idxs = [i for i, s in enumerate(scores) if s >= threshold]
    idxs.sort()

    if len(idxs) < max(1, len(original) // 10):
        top_indices = np.argsort(scores)[::-1][:max(1, len(original) // 5)]
        idxs = sorted(list(set(list(idxs) + list(top_indices))))

    return ' '.join(original[i] for i in idxs)