import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
train_df = pd.read_csv(r'D:\BigDataAnalytics\twitter_training.csv') # Update to your path
validation_df = pd.read_csv(r'D:\BigDataAnalytics\twitter_validation.csv') # Update to your path

# Check the columns and sample data to understand the structure
print(f"Training Data Columns: {train_df.columns}")
print(f"Validation Data Columns: {validation_df.columns}")
print(f"Training Data Sample:\n{train_df.head()}")
print(f"Validation Data Sample:\n{validation_df.head()}")

# Since the last column contains the content, we'll use iloc to get it
X_train = train_df.iloc[:, -1]  # Last column (Content)
y_train = train_df.iloc[:, 2]  # Assuming labels are in the third column (Positive/Negative)

X_val = validation_df.iloc[:, -1]  # Last column (Content)
y_val = validation_df.iloc[:, 2]  # Assuming labels are in the third column (Irrelevant/Relevant)

# Remove NaN values from the 'Content' column and ensure the data is a string
train_df.dropna(subset=[train_df.columns[-1]], inplace=True)
validation_df.dropna(subset=[validation_df.columns[-1]], inplace=True)

# Make sure the text data is actually a string
X_train = X_train.apply(str)
X_val = X_val.apply(str)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=7500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train the model using LinearSVC
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Predict on both training and validation data
y_train_pred = model.predict(X_train_tfidf)
y_val_pred = model.predict(X_val_tfidf)

# Calculate accuracy for both training and validation
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Print accuracy for both datasets
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Print classification report for training and validation data
print("\nClassification Report (Training):")
print(classification_report(y_train, y_train_pred))

print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix for training and validation data
train_cm = confusion_matrix(y_train, y_train_pred)
val_cm = confusion_matrix(y_val, y_val_pred)

# Plot confusion matrix for training data
plt.figure(figsize=(10, 7))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Training Data)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot confusion matrix for validation data
plt.figure(figsize=(10, 7))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix (Validation Data)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
