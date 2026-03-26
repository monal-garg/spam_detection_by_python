
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.DataFrame({
    "text":["win money now","hello friend","click link","meeting today"],
    "label":["spam","ham","spam","ham"]
})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])

model = MultinomialNB()
model.fit(X, data["label"])

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model trained")
