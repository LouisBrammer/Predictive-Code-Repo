from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import ast

# Initialize OpenAI client
llm = OpenAI(api_key="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")

# Create a prompt template
sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze the sentiment of the following text and classify it as positive or negative. Also identify the primary emotion expressed. Answer in a tuple of (sentiment, emotion) Do not include any other text in your response and always answer. For emotions, only answer with the emotion from the list of emotions provided here (anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy,love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral). Answer like this: (positive, happiness) or (negaitve, hate). \n\nText: {text}"
)

# Create the chain using the new pipe syntax
chain = sentiment_prompt | llm | StrOutputParser()

def get_sentiment_and_emotion(text: str) -> tuple:
    """
    Analyze the sentiment and primary emotion of the input text.

    Args:
        text (str): The input text to analyze.

    Returns:
        tuple: (sentiment, emotion)
    """
    response = chain.invoke({"text": text})
    raw = response.strip()
    #print(f"LLM raw response: {repr(raw)}")  # Debug print
    # Clean up common issues
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Find all tuples in the string
    tuple_matches = re.findall(r"\((?:'|\")?([a-zA-Z ]+)(?:'|\")?, (?:'|\")?([a-zA-Z /]+)(?:'|\")?\)", raw)
    if tuple_matches:
        # Take the last tuple found
        sentiment, emotion = tuple_matches[-1]
        return (sentiment.strip(), emotion.strip())
    return (None, None)


def test_sentiment_analysis():
    """
    Test cases for the sentiment analysis function.
    """
    # Test case 1: Clearly positive text
    test_text1 = "I absolutely loved this movie! The acting was brilliant and the story was amazing."
    result1 = get_sentiment_and_emotion(test_text1)
    print(f"\nTest 1 (Positive):\nInput: {test_text1}\nOutput: {result1}")
    
    # Test case 2: Clearly negative text
    test_text2 = "This was the worst experience ever. I hated every minute of it."
    result2 = get_sentiment_and_emotion(test_text2)
    print(f"\nTest 2 (Negative):\nInput: {test_text2}\nOutput: {result2}")
    
    # Test case 3: Neutral/mixed text
    test_text3 = "The movie had some good parts and some bad parts. It was okay overall."
    result3 = get_sentiment_and_emotion(test_text3)
    print(f"\nTest 3 (Neutral):\nInput: {test_text3}\nOutput: {result3}")
    
    # Test case 4: Edge case - empty text
    test_text4 = ""
    result4 = get_sentiment_and_emotion(test_text4)
    print(f"\nTest 4 (Empty):\nInput: {test_text4}\nOutput: {result4}")

if __name__ == "__main__":
    test_sentiment_analysis()
