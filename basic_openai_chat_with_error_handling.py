from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

try:
    # Create a chat completion request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "List five data science professions."}]
    )
    # Print the response
    print("Response from OpenAI:")
    print(response.choices[0].message.content)

except openai.AuthenticationError as e:
    print(f"OpenAI API failed to authenticate: {e}")

except openai.RateLimitError as e:
    print(f"OpenAI API request exceeded rate limit: {e}")

except Exception as e:
    print(f"Unable to generate a response. Exception: {e}")
