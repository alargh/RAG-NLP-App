import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)

    words = [word_tokenize(sent.lower()) for sent in sentences]

    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()

    words_filtered = [
        [lemmatizer.lemmatize(w) for w in sent if w not in stop_words and w not in string.punctuation]
        for sent in words
    ]

    sentences_filtered = [' '.join(ws) for ws in words_filtered]
    return sentences_filtered, sentences