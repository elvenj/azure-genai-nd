import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

# --- 1. SYSTEM MESSAGE EXERCISE ---
# Goal: Test categorization with and without a specific System Message
print("\n--- 1. System Message Exercise ---")

article_prompt = """What kind of article is this?
---
Pittsburgh Pirates Wins against New York Mets
Pittsburgh Pirates mounted a big 6-0 shutout against the Pittsburgh Pirates last night, 
solidifying their win with a 3 run homerun. The Mets three hits came in the 2nd and 
the 5th innings but were unable to get the runner home to score"""

# First attempt: Default behavior
response_default = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": article_prompt}
    ]
)
print(f"Default Response (with explanation): {response_default.choices[0].message.content}")

# Second attempt: Targeted System Message
response_system = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a news aggregator that categorizes news articles. Just give the category. Nothing else."},
        {"role": "user", "content": article_prompt}
    ]
)
print(f"System Message Response (Just category): {response_system.choices[0].message.content}")


# --- 2. FORMAT OUTPUT EXERCISE ---
# Goal: Request outputs in Table and JSON formats
print("\n--- 2. Format Output Exercise ---")

table_prompt = "Create a list of biking trails in New York. Output as table with trail name, distance, difficulty level"
response_table = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": table_prompt}]
)
print(f"Table Output:\n{response_table.choices[0].message.content}")

json_prompt = "Create a list of biking trails in New York. Output as json with trail name, distance, difficulty level"
response_json = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": json_prompt}]
)
print(f"JSON Output:\n{response_json.choices[0].message.content}")


# --- 3. CHAIN OF THOUGHT EXERCISE ---
# Goal: Force the model to reason step-by-step
print("\n--- 3. Chain of Thought Exercise ---")

cot_prompt = """Who was the highest goal scorer in the recent soccer worldcup? 
Take a step-by-step approach in your response, cite sources and give reasoning 
before sharing final answer in the below format: ANSWER is: <name>"""

response_cot = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": cot_prompt}]
)
print(f"Chain of Thought Response:\n{response_cot.choices[0].message.content}")


# --- 4. HYPERPARAMETER TUNING EXERCISE ---
# Goal: Compare outputs with different Temperatures and Top_P values
print("\n--- 4. Hyperparameter Tuning Exercise ---")

ice_cream_prompt = "Generate a 10 word description for a new ice cream shop"

# Testing Temperature
for temp in [0.8, 0.2]:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": ice_cream_prompt}],
        temperature=temp
    )
    print(f"Temperature {temp}: {resp.choices[0].message.content}")

# Testing Top_P
for top_p in [0.9, 0.3]:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": ice_cream_prompt}],
        top_p=top_p
    )
    print(f"Top_P {top_p}: {resp.choices[0].message.content}")