import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from functools import cache
import pickle
import os
from functools import wraps
from typing import Callable, Any
from time import sleep

def retry(retries: int = -1, delay: float = 1) -> Callable:
    """
    Attempt to call a function, if it fails, try again with a specified delay.

    :param retries: The max amount of retries you want for the function call. Use -1 for infinite retries.
    :param delay: The delay (in seconds) between each function retry
    :return:
    """

    # Don't let the user use this decorator if they are high
    if retries == 0 or delay <= 0:
        raise ValueError('Are you high, mate?')

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            i = 1
            while True:  
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Break out of the loop if the max amount of retries is exceeded
                    if retries != -1 and i > retries:
                        print(f'Error: {repr(e)}.')
                        print(f'"{func.__name__}()" failed after {i} retries.')
                        break
                    else:
                        print(f'Error: {repr(e)} -> Retrying...')
                        sleep(delay)  # Add a delay before running the next iteration
                        i += 1

        return wrapper

    return decorator

# Sample text data for demonstration
train_data = ["movie good", "I like this movie", "I love this movie", "This movie is great", "I dislike this movie", "This movie is terrible", "I hate this movie", "movie bad"]
train_labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Define the text classification model
@cache
def create_model():
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    return model

# Train the model
@cache
def train_model(train_data, train_labels):
    model = create_model()
    model.fit(train_data, train_labels)
    return model

# Function to make predictions on user input text
@cache
def predict_text(model, text):
    try:
        prediction = model.predict([text])
        return prediction[0]
    except ValueError:
        return -1  # Return -1 for unknown sentiment

# Check if the saved model exists, and if yes, load it
if os.path.exists("text_classification_model.pkl"):
    with open("text_classification_model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    # Train the model if the saved model does not exist
    model = train_model(tuple(train_data), tuple(train_labels))
    # Save the trained model to disk
    with open("text_classification_model.pkl", "wb") as f:
        pickle.dump(model, f)
        print("Model made")

print("---Known stuff---")
for data in train_data:
    print(data)
print("--------------------")

@retry()
def ask():
    user_input = input("Enter text: ")
    prediction = predict_text(model, user_input)
    # Display the prediction
    if prediction == 1:
        print("Prediction: Positive")
    elif prediction == 0:
        print("Prediction: Negative")
    else:
        raise ValueError("Unknown prediction")

# Call the ask function to start the interaction
ask()
