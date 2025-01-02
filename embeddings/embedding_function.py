import os
import json

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# Define a create_embeddings function
def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    response_dict = response.model_dump()
  
    return [data['embedding'] for data in response_dict['data']]

print(create_embeddings(["Python is the best!", "R is the best!"]))
print(create_embeddings("DataCamp is awesome!")[0])