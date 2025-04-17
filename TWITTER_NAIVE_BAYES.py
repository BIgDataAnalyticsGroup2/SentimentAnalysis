import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
train_file = r'D:\BigDataAnalytics\twitter_training.csv' # Update to your path
validation_file = r'D:\BigDataAnalytics\twitter_validation.csv' # Update to your path

# Load the data
data_train = pd.read_csv(train_file)
data_validation = pd.read_csv(validation_file)

# Display column names and sample data to verify structure
print("Training Data Columns:", data_train.columns)
print("Validation Data Columns:", data_validation.columns)

# Rename the columns to match the 4-column format
data_train.columns = ['ID', 'Source', 'Label', 'Content']
data_validation.columns = ['ID', 'Source', 'Label', 'Content']

# Display the first few rows to verify
print("\nTraining Data Sample:")
print(data_train.head())

print("\nValidation Data Sample:")
print(data_validation.head())

# Check for missing values
print("\nMissing Values in Training Data:")
print(data_train.isnull().sum())

print("\nMissing Values in Validation Data:")
print(data_validation.isnull().sum())

# Drop rows with missing 'Content' column values
data_train = data_train.dropna(subset=['Content'])
data_validation = data_validation.dropna(subset=['Content'])

# Display cleaned data
print("\nCleaned Training Data:")
print(data_train.head())

print("\nCleaned Validation Data:")
print(data_validation.head())

# Function to clean the text data
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\[.*?\]', '', text)               # Remove square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text)                # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # Remove punctuation and digits
    text = re.sub(r'\s+', ' ', text).strip()          # Remove extra spaces
    return text

# Apply text cleaning to 'Content' column
data_train['Cleaned_Content'] = data_train['Content'].apply(clean_text)
data_validation['Cleaned_Content'] = data_validation['Content'].apply(clean_text)

# Display cleaned content samples
print("\nCleaned Training Data Sample:")
print(data_train[['Content', 'Cleaned_Content']].head())

print("\nCleaned Validation Data Sample:")
print(data_validation[['Content', 'Cleaned_Content']].head())

# Splitting the training data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data_train['Cleaned_Content'],
    data_train['Label'],
    test_size=0.2,
    random_state=42
)

# Validation dataset
X_val = data_validation['Cleaned_Content']
y_val = data_validation['Label']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
X_val_tfidf = vectorizer.transform(X_val)

print("\nTF-IDF Vectorizer Shape:")
print("X_train:", X_train_tfidf.shape)
print("X_test:", X_test_tfidf.shape)
print("X_val:", X_val_tfidf.shape)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred_train = model.predict(X_train_tfidf)
y_pred_test = model.predict(X_test_tfidf)
y_pred_val = model.predict(X_val_tfidf)

# Model evaluation
print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))

# Classification report
print("\nClassification Report (Validation Data):")
print(classification_report(y_val, y_pred_val))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_val)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Validation Data)')
plt.show()


# Test with new sentences
test_sentences = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst experience ever. Completely disappointed.",
    "I'm feeling okay about it. Not great, but not bad either."
]

# Clean and vectorize the new inputs
test_sentences_clean = [clean_text(sentence) for sentence in test_sentences]
test_sentences_tfidf = vectorizer.transform(test_sentences_clean)

# Predict sentiment
predictions = model.predict(test_sentences_tfidf)

# Display results
for i, sentence in enumerate(test_sentences):
    print(f"\nInput: {sentence}")
    print(f"Predicted Sentiment: {predictions[i]}")

