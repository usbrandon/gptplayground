import os
import json
from pprint import pprint

import openai
from dotenv import load_dotenv
from openai import OpenAI


def preview_embedding(data, max_items=2, vector_preview_length=5):
    """
    Creates a readable preview of embeddings or nested data structures.
    
    Args:
        data: List or dict containing the data
        max_items: Number of items to show from lists
        vector_preview_length: Number of elements to show from long vectors/lists
    """
    if isinstance(data, list):
        if len(data) > max_items:
            preview_data = data[:max_items]
            remaining = len(data) - max_items
            preview_data.append(f"... and {remaining} more items")
        else:
            preview_data = data
            
        # Handle each item
        for i, item in enumerate(preview_data):
            if isinstance(item, (list, dict)):
                preview_data[i] = preview_embedding(item, max_items, vector_preview_length)
            elif isinstance(item, str) and len(item) > 100:  # For long strings
                preview_data[i] = item[:100] + "..."
                
        return preview_data
    
    elif isinstance(data, dict):
        return {k: preview_embedding(v, max_items, vector_preview_length) 
                if isinstance(v, (list, dict)) 
                else v for k, v in data.items()}


def preview_vector(vector, preview_length=5):
    """Preview a numerical vector/array"""
    if len(vector) > preview_length * 2:
        return (
            f"[{', '.join(f'{x:.3f}' for x in vector[:preview_length])}, ..., "
            f"{', '.join(f'{x:.3f}' for x in vector[-preview_length:])}]"
        )
    return f"[{', '.join(f'{x:.3f}' for x in vector)}]"


# Load environment variables from .env file
load_dotenv()

#
# Huge products list with embeddings from DataCamp's: https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/what-are-embeddings?ex=7
#
with open('products.json', 'r') as file:
    products = json.load(file)

# Generate a preview of the embedding to understand its structure.
preview = preview_embedding(products)
pprint(preview, width=80, indent=2)

# Set OpenAI API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
# Extract a list of product short descriptions from products
product_descriptions = [product['short_description'] for product in products]

# Create embeddings for each product description
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=product_descriptions
)
response_dict = response.model_dump()

# Extract the embeddings from response_dict and store in products
for i, product in enumerate(products):
    product['embedding'] = response_dict['data'][i]['embedding']
    
#print(products[0].items())
print(type(products))