import os
import json

import chromadb
from chromadb.utils import embedding_functions
import tiktoken

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# Create a persistant client
client = chromadb.PersistentClient()

if "netflix_titles" in [c for c in client.list_collections()]:
    collection = client.get_collection("netflix_titles")
else:
    collection = client.create_collection(
    # Create a netflix_title collection using the OpenAI Embedding function    
        name="netflix_titles",
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=os.getenv('OPENAI_API_KEY'))
    )

# List the collections
print(client.list_collections())

file_path = "./netflix_titles.json"

with open(file_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Inspect the first document to see its structure.
print(documents[0])

# Load the encoder for the OpenAI text-embedding-3-small model
enc = tiktoken.encoding_for_model("text-embedding-3-small")

# Encode each text in documents and calculate the total tokens
total_tokens = sum(len(enc.encode(doc["document"])) for doc in documents)

cost_per_1k_tokens = 0.00002

# Display number of tokens and cost
print('Total tokens:', total_tokens)
print('Cost:', cost_per_1k_tokens * total_tokens/1000)


# Add the documents to the collection
# Assume you have:
#  - `collection` already created or retrieved
#  - `ids` is a list of unique IDs
#  - `documents` is a list of strings containing text data

# Add documents if they’re not in the collection yet
new_count = 0
skipped_count = 0

for doc in documents:
    doc_id = doc["id"]
    doc_text = doc["document"]
    
    existing = collection.get(ids=[doc_id])
    if len(existing["ids"]) == 0:
        collection.add(documents=[doc_text], ids=[doc_id])
        new_count += 1
    else:
        skipped_count += 1

print(f"Added {new_count} new documents. Skipped {skipped_count} existing documents.")
print(f"Collection now has {collection.count()} total documents.")
