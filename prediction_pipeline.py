
from tensorflow.keras.preprocessing.sequence import pad_sequences



def prediction_pipeline(text, model, tokenizer, max_len):
    """
    Pipeline function that handles all preprocessing steps and returns the sentiment.
    
    Args:
        text (str): Input text to predict
        model: Trained model
    Returns:
        str: Either "positive" or "negative" sentiment
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded, verbose=0)[0][0]
    return "positive" if prediction > 0.5 else "negative"



if __name__ == "__main__":
    # Example/test code here
    # Example usage                                                                 
    text = "This movie was fantastic! I loved it."
    sentiment = prediction_pipeline(text, model, tokenizer, max_len)
    print(f"Sentiment: {sentiment}")     