import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import Dataset, DatasetDict
import torch
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Load the data
train_df = pd.read_csv(r'D:\BigDataAnalytics\twitter_training.csv') # Update to your path
validation_df = pd.read_csv(r'D:\BigDataAnalytics\twitter_validation.csv') # Update to your path

# Rename columns
train_df.columns = ['ID', 'Source', 'Label', 'Content']
validation_df.columns = ['ID', 'Source', 'Label', 'Content']

# Strip any whitespace from the column names
train_df = train_df[train_df['Content'].notna() & (train_df['Content'].str.strip() != "")]
validation_df = validation_df[validation_df['Content'].notna() & (validation_df['Content'].str.strip() != "")]

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

# Initialize the BERT tokenizer (make sure this is done before the function)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize function (batched=True)
def tokenize_function(examples):
    # Ensure 'examples['Content']' is a list of strings
    content = examples['Content']
    if isinstance(content, str):
        content = [content]  # If it's a single string, turn it into a list

    # Tokenize the batch of texts
    return tokenizer(content, truncation=True, padding='max_length', max_length=128)


# Tokenize the datasets in batches
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)

# Check the tokenized output
print(tokenized_train_dataset[0])

# ✅ 2. Encode labels
label_encoder = LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
validation_df['Label'] = label_encoder.transform(validation_df['Label'])

# ✅ 3. Convert to Hugging Face Datasets format
train_dataset = Dataset.from_pandas(train_df[['Content', 'Label']])
val_dataset = Dataset.from_pandas(validation_df[['Content', 'Label']])
dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})

# ✅ 4. Tokenize using BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example['Content'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("Label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ✅ 5. Load BERT for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# ✅ 6. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log every 10 steps
)

# ✅ 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=lambda eval_pred: {
        "accuracy": accuracy_score(eval_pred.label_ids, torch.argmax(torch.tensor(eval_pred.predictions), axis=1))
    }
)

# ✅ 8. Train the model
trainer.train()

# ✅ 9. Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# ✅ 10. Detailed Classification Report
predictions = trainer.predict(tokenized_dataset["validation"])
preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
print(classification_report(predictions.label_ids, preds, target_names=label_encoder.classes_))

# ✅ 11. Predict new text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    return label_encoder.inverse_transform([predicted_class])[0]

# 🔍 Test predictions
print("\nInput: I love this product! It's amazing and works perfectly.")
print("Predicted Sentiment:", predict_sentiment("I love this product! It's amazing and works perfectly."))

print("\nInput: This is the worst experience ever. Completely disappointed.")
print("Predicted Sentiment:", predict_sentiment("This is the worst experience ever. Completely disappointed."))

print("\nInput: I'm feeling okay about it. Not great, but not bad either.")
print("Predicted Sentiment:", predict_sentiment("I'm feeling okay about it. Not great, but not bad either."))
