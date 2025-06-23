"""
Boss Message Sentiment Analyzer

This script implements a sentiment analysis system specifically designed to interpret your boss's
cryptic messages and determine their mood - whether they're still angry or showing signs of forgiveness.
Understanding their tone will reveal if you are permitted to proceed with the reexamination.

The system uses:
- Text preprocessing techniques to clean the messages
- TF-IDF vectorization to convert text to numerical features
- Multinomial Naive Bayes classifier to categorize messages as 'angry' or 'forgiving'

Your future depends on getting this right!
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Uncomment these lines if running for the first time to download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# ------------------ Model Loading/Initialization ------------------

# Define paths for saving/loading the model and vectorizer
model_path = "boss_sentiment_model.pkl"
vectorizer_path = "boss_sentiment_vectorizer.pkl"

# Check if a pre-trained model exists and load it, or initialize new ones
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Loaded existing boss sentiment model and vectorizer.")
else:
    # Initialize new model and vectorizer
    model = MultinomialNB()
    vectorizer = TfidfVectorizer(max_features=3000, min_df=2)
    print("No existing model found. Training from scratch.")

# ------------------ Text Preprocessing Function ------------------

def preprocess_text(text):
    """
    Preprocess boss's cryptic messages for sentiment analysis.
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Tokenize into individual words
    4. Remove common stopwords
    5. Join back into a single string
    
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

# ------------------ Sample Data ------------------

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

# Apply text preprocessing to both training and prediction datasets
print("Preprocessing boss's messages...")
df_train["cleaned_message"] = df_train["message"].apply(preprocess_text)
df_predict["cleaned_message"] = df_predict["message"].apply(preprocess_text)

# Transform text data into TF-IDF features
X_train = vectorizer.fit_transform(df_train["cleaned_message"])
y_train = df_train["sentiment"]
X_predict = vectorizer.transform(df_predict["cleaned_message"])

print(f"Feature matrix shape: {X_train.shape}")

# ------------------ Model Training or Prediction ------------------

# Ask user whether to train or use the model for prediction
mode = input("Enter 'train' to train the model or 'analyze' to interpret your boss's messages: ").strip().lower()

if mode == "train":
    # Training mode: Split data, train model, evaluate, and save
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training on {X_train_split.shape[0]} examples, testing on {X_test_split.shape[0]} examples")
    
    # Train the Naive Bayes model
    model.fit(X_train_split, y_train_split)
    
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
    
    # Display most important features for each class
    feature_names = vectorizer.get_feature_names_out()
    
    # Top forgiving features
    forgiving_index = np.where(model.classes_ == 'forgiving')[0][0]
    angry_index = np.where(model.classes_ == 'angry')[0][0]
    
    forgiving_coef = model.feature_log_prob_[forgiving_index] - model.feature_log_prob_[angry_index]
    top_forgiving = np.argsort(forgiving_coef)[-10:]
    print("\nTop terms indicating forgiveness:")
    for i in top_forgiving:
        print(f"- {feature_names[i]}")
    
    # Top angry features
    angry_coef = model.feature_log_prob_[angry_index] - model.feature_log_prob_[forgiving_index]
    top_angry = np.argsort(angry_coef)[-10:]
    print("\nTop terms indicating anger:")
    for i in top_angry:
        print(f"- {feature_names[i]}")
    
    # Save the trained model and vectorizer for future use
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved successfully.")

elif mode == "analyze":
    # Prediction mode: Use trained model to analyze boss's mood
    
    # Check if the model and vectorizer files exist
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        # Load the trained model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Make predictions on the test data
        predictions = model.predict(X_predict)
        
        # Get probability scores for each prediction
        probabilities = model.predict_proba(X_predict)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Add predictions and confidence scores to the DataFrame
        df_predict["mood"] = predictions
        df_predict["confidence"] = confidence_scores
        
        # Display results in a nicely formatted table
        print("\n=== Boss Mood Analysis ===")
        result_df = df_predict[["message", "mood", "confidence"]]
        pd.set_option('display.max_colwidth', 100)  # Limit display width
        print(result_df)
        
        # Count and display mood distribution
        mood_counts = result_df["mood"].value_counts()
        forgiveness_count = mood_counts.get('forgiving', 0)
        angry_count = mood_counts.get('angry', 0)
        total_count = forgiveness_count + angry_count
        
        print(f"\nMood Analysis Summary:")
        print(f"Forgiveness indicators: {forgiveness_count}/{total_count} messages ({forgiveness_count/total_count*100:.1f}%)")
        print(f"Anger indicators: {angry_count}/{total_count} messages ({angry_count/total_count*100:.1f}%)")
        
        # Provide an overall assessment
        if forgiveness_count > angry_count:
            print("\nOverall Assessment: Your boss appears to be showing signs of forgiveness.")
            print("Recommendation: This may be a good time to approach them about the reexamination.")
        else:
            print("\nOverall Assessment: Your boss still seems to be angry or frustrated.")
            print("Recommendation: It might be better to wait before requesting the reexamination.")
        
        # Ask if user wants to analyze a custom message from their boss
        custom_analysis = input("\nDid your boss send a new message? Would you like to analyze it? (yes/no): ").strip().lower()
        if custom_analysis == "yes":
            while True:
                custom_msg = input("\nEnter your boss's message (or 'quit' to exit): ")
                if custom_msg.lower() == 'quit':
                    break
                
                # Preprocess and predict
                cleaned_msg = preprocess_text(custom_msg)
                msg_vector = vectorizer.transform([cleaned_msg])
                mood = model.predict(msg_vector)[0]
                confidence = np.max(model.predict_proba(msg_vector))
                
                print(f"Boss's Mood: {mood.upper()} (Confidence: {confidence:.2f})")
                
                if mood == 'forgiving':
                    print("This message suggests your boss may be open to discussing the reexamination.")
                else:
                    print("This message suggests your boss is still upset. It might be best to wait.")
    else:
        print("No trained model found. Please train the model first.")

else:
    # Invalid input handling
    print("Invalid input. Please enter 'train' or 'analyze'.")