import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load configurations
load_dotenv()

client = AzureOpenAI(
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_TEXT_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_TEXT_KEY")
)

def run_rag_inference():
    """
    Executes a RAG query against the uploaded Travel Brochures.
    """
    
    # 1. Setup System Message (The Persona)
    system_message = """I am a travel enthusiast named Go who helps people discover their travel destinations. 
I am upbeat and friendly. I introduce myself when first 
saying hello. When helping people out, I always ask them 
for this information to inform the travel recommendation 
I provide:
1. What are their travel preferences
2. What are there accommodation choices
I will then provide all the suggestions I could find. I will 
also ask the follow up question to find their best travel 
destination."""

    # 2. Udacity Prompt 1
    user_query = "I am looking to travel to a city that has a lot of hustle and bustle. Any recommendations?"
    
    print(f"Connecting to Travel Index: {os.getenv('AZURE_SEARCH_INDEX')}...")

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT"),
            temperature=0.30,  # Required by Lab
            top_p=0.95,        # Required by Lab
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ],
            extra_body={
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
                            "index_name": os.getenv("AZURE_SEARCH_INDEX"),
                            "authentication": {
                                "type": "api_key",
                                "key": os.getenv("AZURE_SEARCH_KEY")
                            }
                        }
                    }
                ]
            }
        )

        # Output the Grounded AI Response
        print("\n--- Travel Copilot 'Go' Response ---")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\n[DEBUG] Error Details: {e}")

if __name__ == "__main__":
    run_rag_inference()