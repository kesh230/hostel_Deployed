# backend/train_model.py
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load the dataset
data = pd.read_csv('f.csv')  # ensure that f.csv has Review (Text) and Sentiment Score columns
print("Dataset loaded successfully.")

# Step 2: Extract features and labels
X = data['Review (Text)']
y = data['Sentiment Score']

# Step 3: Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
print("Text vectorization completed.")

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 5: Train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)
print("Model training completed.")

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Step 7: Save the model and vectorizer
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("Model and vectorizer saved successfully.")
