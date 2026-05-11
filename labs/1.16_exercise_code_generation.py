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

def code_lab_request(task_name, prompt):
    """
    Utility to send coding tasks to gpt-4o and print results.
    """
    print(f"\n{'='*20}")
    print(f"TASK: {task_name}")
    print(f"{'='*20}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert software engineer and polyglot programmer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 # Lower temperature for more precise code generation
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

# --- 1. Generate Python code to print even numbers ---
gen_code_prompt = "Write a Python script to print even numbers between 0 and 20."
code_lab_request("Generate Even Numbers Code", gen_code_prompt)


# --- 2. Fix bug in the python code ---
# The buggy code provided in the lab instructions
buggy_code = """
for num in range(2, 50): 
    # check if the number is prime  
    for i in range(2, 50):             
        if (num % i) == 0:                 
            break             
        else                 
            print(num)
"""

fix_bug_prompt = f"""
The following code is supposed to print prime numbers up to 100 but it only goes to 50 and has syntax/logic errors. 
Please fix it so it correctly prints prime numbers up to 100:

{buggy_code}
"""
code_lab_request("Fix Prime Numbers Bug", fix_bug_prompt)


# --- 3. Generate Documentation ---
# We'll ask the model to document the corrected version of the prime number logic
doc_prompt = "Generate detailed technical documentation for a Python script that calculates and prints prime numbers up to 100."
code_lab_request("Generate Documentation", doc_prompt)


# --- 4. Translate Code to C# ---
translate_prompt = "Translate the fixed Python code for printing prime numbers up to 100 into C#."
code_lab_request("Translate Python to C#", translate_prompt)