import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# Download stopwords only once
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


def remove_html(text):
    """Remove HTML tags, scripts, CSS."""
    soup = BeautifulSoup(text, "html.parser")

    # Remove script & style tags
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator=" ")


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_symbols(text):
    """Remove special characters and unnecessary symbols."""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text


def normalize_text(text):
    """Normalize text - lowercase, clean multiple spaces."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    """Remove English stopwords."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def preprocess_text(text):
    """Full text cleaning pipeline used before prediction/training."""
    
    if not text:
        return ""

    text = remove_html(text)
    text = remove_urls(text)
    text = remove_symbols(text)
    text = normalize_text(text)
    text = remove_stopwords(text)

    return text
