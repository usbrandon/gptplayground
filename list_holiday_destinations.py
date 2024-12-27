"""
OpenAI API interaction module with retry functionality.

This module provides a robust interface for making requests to the OpenAI API
with exponential backoff retry logic to handle transient failures. It uses
environment variables for secure API key management.

Dependencies:
    - tenacity: For retry logic
    - openai: For API interaction
    - python-dotenv: For environment variable management

Author: Brandon Jackson
Version: 1.0.0
"""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(model: str, message: dict) -> str:
    """
    Send a request to OpenAI API and get the response with retry logic.

    This function sends a single message to the specified OpenAI model and retrieves
    the response. It implements exponential backoff retry logic to handle temporary
    failures and rate limits.

    Args:
        model (str): The OpenAI model identifier to use (e.g., "gpt-4", "gpt-3.5-turbo")
        message (dict): A dictionary containing the message with 'role' and 'content' keys
                       Example: {"role": "user", "content": "Hello, world!"}

    Returns:
        str: The content of the model's response message

    Raises:
        TenacityError: If all retry attempts fail after 6 attempts
        OpenAIError: For API-specific errors not handled by retry logic
        ValueError: If the message dictionary is missing required keys

    Example:
        >>> message = {"role": "user", "content": "List ten holiday destinations."}
        >>> response = get_response("gpt-4", message)
        >>> print(response)
    """
    response = client.chat.completions.create(
        model=model,
        messages=[message]
    )
    return response.choices[0].message.content

print(get_response("gpt-4o-mini", {"role": "user", "content": "List ten holiday destinations."}))