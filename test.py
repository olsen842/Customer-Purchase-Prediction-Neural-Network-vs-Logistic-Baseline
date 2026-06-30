import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# Load data
df = pd.read_csv('fake_data.csv')

X = df.drop('label', axis=1).values
y = df['label'].values


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model (Logistic Regression)
model = LogisticRegression()


# Train model
model.fit(X_train, y_train)


# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]


# Apply custom threshold (0.6 like your NN)
preds = (probs > 0.6).astype(int)


# Metrics
accuracy = accuracy_score(y_test, preds)

n_buy = preds.sum()
n_skip = (preds == 0).sum()


print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Number of buyers predicted: {n_buy}")
print(f"Number of skips predicted: {n_skip}")

print("\nClassification Report:")
print(classification_report(y_test, preds))