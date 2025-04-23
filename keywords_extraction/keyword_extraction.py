import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from src.setup_nltk import setup_nltk

ESSENTIAL_KEYWORDS = [
    "concert", "tour", "venue", "performance", "artist", "band", "musician",
    "stage", "ticket", "festival", "audience", "setlist", "backstage",
    "headliner", "opening_act", "acoustic", "amplifier", "arena",
    "amphitheater", "stadium", "theater", "pavilion", "hall", "seating",
    "standing_room", "vip", "merchandise", "soundcheck", "road_crew",
    "promoter", "lighting", "sound_system", "mixing_board", "monitor",
    "speaker_system", "microphone", "guitar", "bass", "drum_kit",
    "keyboard", "vocal", "singer", "live_performance", "unplugged",
    "2025", "2026"
]

STOP_WORDS = set(stopwords.words('english'))

def extract_artist_names(text):
    pattern = r"\d+\.\s+\*\*(.*?)\*\*"
    matches = re.findall(pattern, text)
    parts = []
    for name in matches:
        cleaned = re.sub(r"[^\w\s]", "", name).lower().strip()
        for word in cleaned.split():
            if word.isalpha() and len(word) > 2 and word not in STOP_WORDS:
                parts.append(word)
    return list(set(parts))

def extract_keywords_nltk(text, top_n=150):
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    tokens = word_tokenize(clean_text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]

    freq_dist = FreqDist(tokens)
    most_common = freq_dist.most_common(top_n)
    return [word for word, _ in most_common]

def extract_and_save_keywords(input_path, output_path, top_n=150):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    artist_keywords = extract_artist_names(text)
    top_keywords = extract_keywords_nltk(text, top_n)

    combined = set(ESSENTIAL_KEYWORDS) | set(artist_keywords) | set(top_keywords)
    sorted_keywords = sorted(combined)

    with open(output_path, 'w', encoding='utf-8') as f:
        for kw in sorted_keywords:
            f.write(f"{kw}\n")

    print(f"Saved {len(sorted_keywords)} keywords to {output_path}")

if __name__ == '__main__':
    setup_nltk()
    extract_and_save_keywords(
        input_path='tour_concert_extraction_file.txt',
        output_path='keywords.txt',
        top_n=150
    )
