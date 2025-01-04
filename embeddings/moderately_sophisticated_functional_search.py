import os
import json

from openai import OpenAI
from dotenv import load_dotenv
from scipy.spatial import distance

load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# 1. Load products from products.json
def load_products(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 2. Combine product features into a single string
def create_product_text(product):
    return f"""Title: {product['title']}
Description: {product['short_description']}
Category: {product['category']}
Features: {', '.join(product['features'])}"""

# 3. Generate embeddings
def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    response_dict = response.model_dump()
    return [data['embedding'] for data in response_dict['data']]

# 4. Find the N closest products using cosine distance
def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    return distances_sorted[:n]

# 5. Main function
def main():
    # Load products
    products = load_products('products.json')
    
    # Create combined product texts
    product_texts = [create_product_text(product) for product in products]
    
    # Generate embeddings for products
    product_embeddings = create_embeddings(product_texts)
    
    # Query
    query_text = "computer"
    query_vector = create_embeddings([query_text])[0]
    
    # Find closest products
    hits = find_n_closest(query_vector, product_embeddings, n=5)
    
    print(f'Search results for "{query_text}":')
    for hit in hits:
        product = products[hit['index']]
        print(f"- {product['title']}")

if __name__ == '__main__':
    main()
