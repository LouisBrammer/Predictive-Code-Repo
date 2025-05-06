from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize OpenAI client
llm = OpenAI(api_key="your_api_key")

# Create a prompt template
sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze the sentiment of the following text and classify it as positive, negative, or neutral. Also identify the primary emotion expressed.\n\nText: {text}"
)

# Create a chain
chain = LLMChain(llm=llm, prompt=sentiment_prompt)

# Example usage
def analyze_sentiment(text):
    response = chain.run(text=text)
    return response

if __name__ == "__main__":
    # Example text for testing
    test_text = "I'm really excited about this new project! It's going to be amazing."
    result = analyze_sentiment(test_text)
    print(result) 