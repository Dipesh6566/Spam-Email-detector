# train.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')

# 2. Keep only the useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 3. Convert spam/ham to numbers (spam=1, ham=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 5. Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Check accuracy
predictions = model.predict(X_test_vec)
print(f"✅ Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 8. Save the model and vectorizer
with open('model/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved successfully!")

