"""
preprocess.py
Purpose: text-cleaning helper used by BOTH train_model.py and app.py.
Using the same function in both places guarantees that training text and
user-entered text are cleaned in exactly the same way.
"""
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # built-in list, no download needed


def clean_text(text):
    """Lowercase, remove URLs, numbers, punctuation and stopwords."""
    text = str(text).lower()                                   # 1. lowercase
    text = re.sub(r"http\S+|www\.\S+", " url ", text)          # 2. replace links with the word 'url'
    text = re.sub(r"\d+", " ", text)                           # 3. remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # 4. remove punctuation
    text = re.sub(r"[^a-z\s]", " ", text)                      # 5. remove any other symbols
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]  # 6. remove stopwords
    return " ".join(words)


if __name__ == "__main__":
    # Quick self-test: python preprocess.py
    sample = "WINNER!! Claim your $1000 prize at http://bit.ly/abc NOW!!!"
    print("Before:", sample)
    print("After :", clean_text(sample))
