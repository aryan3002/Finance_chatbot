import json, random, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

random.seed(42); np.random.seed(42)

# Create a tiny seed dataset you can expand later
seed = [
  {"text":"Customers must provide government-issued ID and proof of address before account opening.","label":"KYC"},
  {"text":"Institutions should monitor transactions for structuring and report suspicious activity.","label":"AML"},
  {"text":"Screen all counterparties against the OFAC sanctions list prior to processing payments.","label":"Sanctions"},
  {"text":"A Suspicious Activity Report must be filed within 30 days of initial detection.","label":"SAR"},
  {"text":"Encrypt cardholder data in transit and at rest as per PCI DSS.","label":"PCI"},
  {"text":"Collect only necessary personal data and honor deletion requests under privacy laws.","label":"Privacy"},
  # add more rows as you encounter real text
]

df = pd.DataFrame(seed)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, stratify=df["label"], random_state=42)

clf = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)),
    ("svm", LinearSVC())
])
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))

# Save model
import joblib, os
Path("artifacts").mkdir(exist_ok=True, parents=True)
joblib.dump(clf, "artifacts/topic_classifier.joblib")
print("Saved artifacts/topic_classifier.joblib")
