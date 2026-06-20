import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# dataset
df = pd.read_csv("data.csv")

# features & labels
X = df["text"]
y = df["label"]

# text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# model
model = LogisticRegression()
model.fit(X_vec, y)

# save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vector.pkl", "wb"))

print("Model trained successfully 🚀")