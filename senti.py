import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import spacy


# Sample dataset (you can replace with Amazon/Flipkart review dataset)
data = {
    "review": [
        "The delivery was fast and packaging was great!",
        "Product quality is poor and not worth the price.",
        "Amazing phone! Battery life is excellent.",
        "Terrible service, I will never order again.",
        "Good value for money. Camera is decent."
    ],
    "sentiment": ["positive", "negative", "positive", "negative", "positive"]
}
df = pd.DataFrame(data)

# Preprocessing function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special chars
    return text.lower()

df["clean_review"] = df["review"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["sentiment"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------- Aspect-Based Sentiment (Simple Example) ----------------
nlp = spacy.load("en_core_web_sm")
aspects = ["delivery", "product", "price", "service", "camera", "battery"]

def extract_aspects(text):
    doc = nlp(text)
    found_aspects = [token.text.lower() for token in doc if token.text.lower() in aspects]
    return list(set(found_aspects))

df["aspects"] = df["review"].apply(extract_aspects)

print("\nAspect Extraction Example:\n", df[["review", "aspects"]])