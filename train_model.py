import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load and clean the dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')

# Rename columns to something clearer
df = df.rename(columns={'Category': 'label', 'Message': 'message'})

# Remove duplicate messages
print("Before removing duplicates:", df.shape)
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# Convert labels: ham:0, spam:1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


# Step 2: Clean the message text
def clean_text(text):
    text = text.lower()                          # lowercase everything
    text = re.sub(r'\d+', ' ', text)              # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)          # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra whitespace
    return text


df['clean_message'] = df['message'].apply(clean_text)

# Quick check: print a few examples before and after cleaning
print("\nBefore and after cleaning (first 3 messages):")
for i in range(3):
    print("Original:", df['message'].iloc[i])
    print("Cleaned: ", df['clean_message'].iloc[i])
    print()


# Step 3: Split into train and test sets
X = df['clean_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])
print("\nSpam ratio in train:", y_train.mean())
print("Spam ratio in test:", y_test.mean())


# Step 4: Convert text into numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTF-IDF shape (train):", X_train_vec.shape)
print("TF-IDF shape (test):", X_test_vec.shape)


# Step 5: Train and compare models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

results = []
best_model = None
best_f1 = -1
best_model_name = None

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 score:  {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")


# Step 6: Save the best model and vectorizer
joblib.dump(best_model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print(f"\nSaved {best_model_name} as model.pkl")
print("Saved TF-IDF vectorizer as vectorizer.pkl")