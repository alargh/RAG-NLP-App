import os
import nltk

def setup_nltk():
    DOWNLOAD_DIR = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    nltk.download('punkt', download_dir=DOWNLOAD_DIR, quiet=True)
    nltk.download('punkt_tab', download_dir=DOWNLOAD_DIR, quiet=True)
    nltk.download('stopwords', download_dir=DOWNLOAD_DIR, quiet=True)

    nltk.data.path.append(DOWNLOAD_DIR)

setup_nltk()