from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize OpenAI client
llm = OpenAI(api_key="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")

# Create a prompt template
sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze the sentiment of the following text and classify it as positive, negative, or neutral. Also identify the primary emotion expressed.\n\nText: {text}"
)

# Create the chain using the new pipe syntax
chain = sentiment_prompt | llm | StrOutputParser()

# Example usage
def analyze_sentiment(text):
    response = chain.invoke({"text": text})
    return response

if __name__ == "__main__":
    # Example text for testing
    test_text = "Isn't it great that Anny is moving out?."
    result = analyze_sentiment(test_text)
    print(result) 