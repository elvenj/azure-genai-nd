import os
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv

# Load laboratory environment variables
load_dotenv()

# Infrastructure parameters validated via Azure Portal
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://elven-service.search.windows.net")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "rag-1779023072560"  # Managed index successfully created

def validate_foundry_environment():
    """
    Validates connectivity with Azure AI Search and verifies the existence
    of the vector index generated during Lab 1.8.
    """
    print("=== Validating Azure AI Foundry Environment (Lab 1.8/1.9) ===")
    print(f"Search Endpoint: {SEARCH_ENDPOINT}")
    print(f"Target Index Name: {INDEX_NAME}\n")
    
    # Initialize client using the best available authentication strategy
    if SEARCH_KEY:
        credential = AzureKeyCredential(SEARCH_KEY)
    else:
        print("[Warning] Search API key not found in .env. Falling back to DefaultAzureCredential...")
        credential = DefaultAzureCredential()
        
    client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    
    try:
        print(f"Checking existence of index '{INDEX_NAME}' on Azure cloud...")
        index = client.get_index(INDEX_NAME)
        print("\n[SUCCESS] Cloud infrastructure successfully validated!")
        print(f" -> Index ID: {index.name}")
        
        print("\n -> Structural Fields:")
        for field in index.fields:
            print(f"    - {field.name} ({field.type})")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to connect or locate the target vector index: {e}")
        print("Verify that infrastructure keys and endpoints match your .env configuration.")

if __name__ == "__main__":
    validate_foundry_environment()