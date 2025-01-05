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

# Find the closest embedding given a query vector
def find_closest(query_vector, embeddings):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    return min(distances, key=lambda x: x["distance"])

# Define the sentiments and reviews

sentiments = [{'label': 'Positive',
               'description': 'A positive restaurant review'},
              {'label': 'Neutral',
               'description':'A neutral restaurant review'},
              {'label': 'Negative',
               'description': 'A negative restaurant review'}]

reviews = ["The food was delicious!",
           "The service was a bit slow but the food was good",
           "The food was cold, really disappointing!"]


# Extract and embed the descriptions from sentiments
class_descriptions = [sentiment['description'] for sentiment in sentiments]
class_embeddings = create_embeddings(class_descriptions)
review_embeddings = create_embeddings(reviews)


# Find the closest sentiment for each review
for index, review in enumerate(reviews):
    closest = find_closest(review_embeddings[index], class_embeddings)
    label = sentiments[closest['index']]['label']
    print(f'"{review}" was classified as {label}')