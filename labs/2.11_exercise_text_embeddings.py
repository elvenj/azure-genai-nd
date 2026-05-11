import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Inicialização nativa e simples
client = AzureOpenAI(
    api_version="2023-05-15", # Versão estável do seu endpoint
    azure_endpoint=os.getenv("AZURE_OPENAI_TEXT_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_TEXT_KEY")
)

def run_lab():
    # Texto para converter em vetor (conceitos do seu perfil)
    text_input = "Biomechanical sensor data fusion using machine learning"
    
    print(f"Gerando embeddings para: {text_input}")
    
    try:
        response = client.embeddings.create(
            input=[text_input],
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )

        embedding = response.data[0].embedding
        print(f"Sucesso! Dimensões do vetor: {len(embedding)}")
        print(f"Primeiros 3 valores: {embedding[:3]}")

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    run_lab()