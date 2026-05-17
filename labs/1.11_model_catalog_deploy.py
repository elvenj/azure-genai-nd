import os
import sys
import requests
from dotenv import load_dotenv

# Load real laboratory environment variables
load_dotenv()

BASE_URL = os.getenv("AZURE_OPENAI_TEXT_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_TEXT_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT", "gpt-4o")

def run_real_telco_extraction():
    """
    Executes raw HTTP inference against the validated Azure OpenAI endpoint
    by dynamically formatting the deployment routing.
    """
    print("=== Executing Raw Lab 1.11 Inference ===")
    
    if not BASE_URL or not API_KEY:
        print("[ERROR] Missing AZURE_OPENAI_TEXT_ENDPOINT or AZURE_OPENAI_TEXT_KEY in .env file.")
        sys.exit(1)

    # Clean potential trailing slashes and build the exact inference path
    clean_base = BASE_URL.rstrip("/")
    target_url = f"{clean_base}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-02-15-preview"
    
    print(f"Constructed URL: {target_url}\n")

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    system_instruction = (
        "You're an AI assistant that helps telco company to extract valuable information "
        "from their conversations by creating JSON files for each conversation transcription you receive. "
        "You always try to extract and format as a JSON:\n"
        "1. Customer Name [name]\n"
        "2. Customer Contact Phone [phone]\n"
        "3. Main Topic of the Conversation [topic]\n"
        "4. Customer Sentiment (Neutral, Positive, Negative)[sentiment]\n"
        "5. How the Agent Handled the Conversation [agent_behavior]\n"
        "6. What was the FINAL Outcome of the Conversation [outcome]\n"
        "7. A really brief Summary of the Conversation [summary]\n"
        "Only extract information that you're sure. If you're unsure, write 'Unknown/Not Found' in the JSON file."
    )

    real_transcript = (
        "Agent: Hello, welcome to Telco's customer service. My name is Juan, how can I assist you?\n"
        "Client: Hello, Juan. I'm calling because I'm having issues with my mobile data plan. It's very slow and I can't browse the internet or use my apps.\n"
        "Agent: I'm very sorry for the inconvenience, sir. Could you please tell me your phone number and your full name?\n"
        "Client: Yes, sure. My number is 011-4567-8910 and my name is Martín Pérez.\n"
        "Agent: Thank you, Mr. Pérez. I'm going to check your plan and your data usage. One moment, please.\n"
        "Client: Okay, thank you.\n"
        "Agent: Mr. Pérez, I've reviewed your plan and I see that you have contracted the basic plan of 2 GB of data per month. Is that correct?\n"
        "Client: Yes, that's correct.\n"
        "Agent: Well, I inform you that you have consumed 90% of your data limit and you only have 200 MB available until the end of the month. That's why your browsing speed has been reduced.\n"
        "Client: What? How is that possible? I barely use the internet on my cell phone. I only check my email and my social networks from time to time. I don't watch videos or download large files.\n"
        "Agent: I understand, Mr. Pérez. But keep in mind that some applications consume data in the background, without you realizing it. For example, automatic updates, backups, GPS, etc.\n"
        "Client: Well, but they didn't explain that to me when I contracted the plan. They told me that with 2 GB I would have enough for the whole month. I feel cheated.\n"
        "Agent: I apologize, Mr. Pérez. It was not our intention to deceive you. I offer you a solution: if you want, you can change your plan to a higher one, with more GB of data and higher speed. This way you can enjoy a better browsing experience.\n"
        "Client: And how much would that cost me?\n"
        "Agent: We have a special offer for you. For only 10 pesos more per month, you can access the premium plan of 5 GB of data and 4G speed. Are you interested?\n"
        "Client: Mmm, I don't know. Isn't there another option? Can't you give me more speed without charging me more?\n"
        "Agent: I'm sorry, Mr. Pérez. That's the only option we have available. If you don't change your plan, you'll have to wait until next month to recover your normal speed. Or you can buy an additional data package, but it would be more expensive than changing plans.\n"
        "Client: Well, let me think about it. Can I call later to confirm?\n"
        "Agent: Of course, Mr. Pérez. You can call whenever you want. The number is the same one you dialed now. Is there anything else I can help you with?\n"
        "Client: No, that's all. Thank you for your attention.Agent: Thank you, Mr. Pérez. Have a good day. Goodbye."
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": real_transcript}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(target_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n[SUCCESS] Production execution completed:")
            print(result["choices"][0]["message"]["content"])
        else:
            print(f"\n[ERROR] HTTP Connection failed with status {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\n[ERROR] Request execution failed: {e}")

if __name__ == "__main__":
    run_real_telco_extraction()