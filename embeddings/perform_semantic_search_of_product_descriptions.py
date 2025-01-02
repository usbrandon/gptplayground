import os
import json

import numpy as np
import openai
from dotenv import load_dotenv
from openai import OpenAI
from scipy.spatial import distance

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


# Load the products
products = json.loads(open('products.json', 'r').read())

# Embed the search text
search_text = "soap"
search_embedding = create_embeddings(search_text)[0]

distances = []
for product in products:
  # Compute the cosine distance for each product description
  dist = distance.cosine(search_embedding, product["embedding"])
  distances.append(dist)

# Find and print the most similar product short_description    
min_dist_ind = np.argmin(distances)
print(products[min_dist_ind]['short_description'])