***FCIM.FIA - Fundamentals of Artificial Intelligence***

> **Lab 6:** *Natural Language Processing* \
> **Performed by:** *Dumitru Moraru*, group *FAF-212* \
> **Verified by:** Elena Graur, asist. univ.

Imports and Utils


```python
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
```

# Part 1
A custom dataset was created containing forgiving and angry sentiment examples.


```python
# Training data with example boss messages and their sentiment labels
training_data = {
    "message": [
        # "Angry" boss messages
        "I expected this report on my desk yesterday. What happened?",
        "This performance is simply unacceptable. We need to talk.",
        "Did you even proofread this before sending it to me?",
        "I'm still waiting for that analysis I requested last week.",
        "Your presentation was disorganized and unprofessional.",
        "I don't have time for excuses. I need results.",
        "This is the third time you've missed this deadline.",
        "These numbers don't add up. Fix this immediately.",
        "I'm disappointed by your lack of attention to detail.",
        "Why wasn't I informed about this issue earlier?",
        "The client is furious. How could you let this happen?",
        "Your absence from yesterday's meeting was noted.",
        "This isn't what I asked for at all. Start over.",
        "You need to take more responsibility for your errors.",
        "I can't believe we're still discussing this problem.",
        
        # "Forgiving" boss messages
        "Let's discuss how we can improve this process moving forward.",
        "I see the effort you've put into this project.",
        "Your recent work shows significant improvement.",
        "I appreciate you taking initiative on this task.",
        "We all make mistakes. Let's find a solution together.",
        "I think you're on the right track with this approach.",
        "Your ideas in the meeting yesterday were insightful.",
        "I'd like to hear your thoughts on how to resolve this.",
        "Thanks for staying late to finish the project.",
        "This report is much better than the previous version.",
        "I'm giving you another opportunity to demonstrate your skills.",
        "Let's set up a time to review your progress.",
        "I noticed the extra effort you've been putting in lately.",
        "Your contribution to the team has been valuable.",
        "I believe you have potential to excel in this role."
    ] * 3,  # Repeat to increase dataset size
    
    "sentiment": [
        # Sentiment labels corresponding to the messages above
        "angry", "angry", "angry", "angry", "angry",
        "angry", "angry", "angry", "angry", "angry",
        "angry", "angry", "angry", "angry", "angry",
        
        "forgiving", "forgiving", "forgiving", "forgiving", "forgiving",
        "forgiving", "forgiving", "forgiving", "forgiving", "forgiving",
        "forgiving", "forgiving", "forgiving", "forgiving", "forgiving"
    ] * 3  # Repeat to match the message repetition
}
```

Explanation

* The dataset consists of text messages and their corresponding sentiment labels (either "forgiving" or "angry").
* To ensure a balanced dataset, sentences are repeated four times.

# Part 2
Before training, we need to clean the text.


```python
def preprocess_text(text):
    """
    Preprocess boss's cryptic messages for sentiment analysis.
    
    Args:
        text (str): The input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Handle missing values
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Join tokens back into a string
    return " ".join(tokens)
```

Explanation

1. Convert to lowercase
2. Remove special characters and numbers
3. Tokenize into individual words
4. Remove common stopwords
5. Join back into a single string

# Part 3
Once the text is cleaned, we need to convert it into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency).


```python
# Transform text data into TF-IDF features
X_train = vectorizer.fit_transform(df_train["cleaned_message"])
y_train = df_train["sentiment"]
X_predict = vectorizer.transform(df_predict["cleaned_message"])
```

Explanation

* TF-IDF vectorization converts words into numerical values based on their importance in the dataset.
* The higher the TF-IDF score, the more important the word is for classification.

# Part 4
To train the model effectively, the dataset is split into training (80%) and testing (20%) sets.


```python
# Split data into training and testing sets (80% train, 20% test)
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training on {X_train_split.shape[0]} examples, testing on {X_test_split.shape[0]} examples")

# Train the Naive Bayes model
model.fit(X_train_split, y_train_split)
```

Explanation

* The Naïve Bayes classifier is trained using the processed dataset.
* Multinomial Naïve Bayes is ideal for text-based classification tasks due to its probabilistic nature.

# Part 5
To assess the model’s performance, we use accuracy, precision, recall, and F1-score.


```python
# Evaluate model performance on the test set
y_pred = model.predict(X_test_split)
accuracy = accuracy_score(y_test_split, y_pred)
report = classification_report(y_test_split, y_pred)
cm = confusion_matrix(y_test_split, y_pred)

print("Training completed!")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:")
print(cm)
```

Explanation

* The model’s predictions are compared to actual labels to determine accuracy.
* The classification report provides insights into precision, recall, and F1-score for each sentiment class.

# Part 6
To avoid retraining the model every time, we save it using joblib.


```python
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
```

To load and reuse the model later:


```python
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
```

Explanation

* Saving the trained model allows it to be reused without retraining.
* Joblib is used because it efficiently stores machine learning models.

# Part 7
Once the model is trained, we can use it to predict sentiment for new sentences.


```python
# Test data for checking boss's current mood
prediction_data = {
    "message": [
        "I'm expecting a comprehensive explanation for this failure.",
        "Perhaps we should revisit your approach to this assignment.",
        "I'd like to see your revised plan by tomorrow morning.",
        "Let's schedule a meeting to discuss your performance.",
        "Your recent report showed some interesting insights.",
        "I noticed you've been putting in extra hours lately.",
        "This isn't what I asked for. Do it again.",
        "We need to address these ongoing issues promptly.",
        "I'm giving you one more chance to prove yourself.",
        "Your teamwork on the latest project was commendable."
    ]
}

# ------------------ Data Preparation ------------------

# Convert dictionaries to pandas DataFrames
df_train = pd.DataFrame(training_data)
df_predict = pd.DataFrame(prediction_data)

# Display dataset information
print(f"Training dataset size: {len(df_train)} examples")
print(f"Prediction dataset size: {len(df_predict)} examples")
```

Explanation

* The model processes new sentences and predicts their sentiment.
* The TF-IDF vectorizer transforms the input text, ensuring compatibility with the trained model.

# Conclusions:
This laboratory work successfully implemented sentiment analysis using Natural Language Processing and Machine Learning techniques. The Naïve Bayes classifier trained on TF-IDF-transformed text demonstrated high accuracy in classifying forgiving and angry sentiments.

# Bibliography:

1) https://www.ibm.com/topics/recurrent-neural-networks
2) https://medium.com/@rebeen.jaff/what-is-lstm-introduction-to-long-short-term-memory-66bd3855b9ce
3) https://www.baeldung.com/cs/lstm-vanishing-gradient-prevention
4) https://medium.com/@anishnama20/understanding-gated-recurrent-unit-gru-in-deep-learning-2e54923f3e2
5) https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#:~:text=A%20Sequence%20to%20Sequence%20network%2C%20or%20seq2seq%20network%2C%20or%20Encoder,to%20produce%20an%20output%20sequence.
6) https://medium.com/@rubyabdullah14/building-a-telegram-bot-with-python-a-step-by-step-guide-5ca305bea6c0
