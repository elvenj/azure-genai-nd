import os
import base64
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Using the API version explicitly shown in your Azure Portal curl example
client = AzureOpenAI(
    api_version="2024-02-01", 
    azure_endpoint=os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_IMAGE_KEY")
)

def run_image_generation():
    prompt = "A cinematic shot of a futuristic biomechanical sensor laboratory, 8k resolution, photorealistic."
    output_path = "labs/1.19_output_image.png"

    print(f"Generating image...")

    try:
        # Removed 'response_format' to fix the 400 error.
        # The model will return either a URL or b64_json based on the region's default.
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1
        )

        image_obj = response.data[0]

        # Robust check: handle both Base64 and URL responses
        if image_obj.b64_json:
            print("Image received as Base64.")
            image_content = base64.b64decode(image_obj.b64_json)
        elif image_obj.url:
            print(f"Image received as URL: {image_obj.url}")
            image_content = requests.get(image_obj.url).content
        else:
            raise Exception("No image data found in the response.")

        # Save to disk
        with open(output_path, "wb") as f:
            f.write(image_content)
            
        print(f"Success! Image saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_image_generation()