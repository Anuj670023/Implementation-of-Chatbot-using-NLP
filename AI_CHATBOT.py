import os
import ssl
import random
import nltk
import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Handling SSL context for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('wordnet')

# Data Preprocessing
def preprocess_text(text):
    tokenizer = nltk.word_tokenize
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = tokenizer(text.lower())  # Tokenize and convert to lowercase
    return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])  # Lemmatize and remove non-alphanumeric tokens

# Load intents from file
with open('DATA.txt', 'r') as file:
    intents = json.load(file)

# Preparing data
patterns, labels = [], []
responses = {}
for intent in intents:
    tag = intent['tag']
    for pattern in intent['patterns']:
        patterns.append(preprocess_text(pattern))
        labels.append(tag)
    responses[tag] = intent['responses']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(patterns, labels, test_size=0.2, random_state=42)

# Vectorizing text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Building the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluating the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

# Displaying results in Streamlit
st.title("Chatbot with Intent Recognition")
st.write("A chatbot powered by NLP and Logistic Regression.")

# User input and prediction
user_input = st.text_input("Enter your message:")
if user_input:
    preprocessed_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([preprocessed_input])
    predicted_intent = model.predict(vectorized_input)[0]
    
    # Generate response
    response = random.choice(responses.get(predicted_intent, ["Sorry, I didn't understand that."]))
    st.write(f"Chatbot: {response}")

# Display model performance
st.write("### Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text(classification_report(y_test, y_pred))
