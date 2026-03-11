import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = "".join([char for char in text if char not in string.punctuation])

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [stemmer.stem(w) for w in words]

    return " ".join(words)