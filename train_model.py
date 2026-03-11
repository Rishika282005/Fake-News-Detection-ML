import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

from preprocessing import clean_text

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

print("Cleaning text...")

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

print("Vectorizing text...")

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = LogisticRegression()
model.fit(X_train, y_train)

print("Saving model...")

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model saved successfully!")