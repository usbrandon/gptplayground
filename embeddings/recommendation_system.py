import os
import json

from openai import OpenAI
from dotenv import load_dotenv
from scipy.spatial import distance
import numpy as np

load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))


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
# Load products and user history
    with open('./embeddings/products.json', 'r') as f:
        products = json.load(f)
    
    with open('./embeddings/user_history.json', 'r') as f:
        history = json.load(f)
    
    # Validate structure
    if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
        raise ValueError("user_history.json must contain a list of dictionaries.")
    
    if not isinstance(products, list) or not all(isinstance(item, dict) for item in products):
        raise ValueError("products.json must contain a list of dictionaries.")
    
    # Create combined product texts
    product_texts = [create_product_text(product) for product in products]
    user_history_texts = [create_product_text(entry) for entry in history]

    # Generate embeddings for products
    product_embeddings = create_embeddings(product_texts)

    # Prepare and embed the user_history, and calculate the mean embeddings
    history_embeddings = create_embeddings(user_history_texts)
    mean_history_embeddings = np.mean(history_embeddings, axis=0)

    # Filter products to remove any in user_history
    products_filtered = [product for product in products if product not in history]

    # Combine product features and embed the resulting texts
    product_texts = [create_product_text(product) for product in products_filtered]
    product_embeddings = create_embeddings(product_texts)

    hits = find_n_closest(mean_history_embeddings, product_embeddings)

    for hit in hits:
        product = products_filtered[hit['index']]
        print(product['title'])

if __name__ == '__main__':
    main()




