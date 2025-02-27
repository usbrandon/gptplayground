"""
OpenAI API interaction module with retry and error handling functionality.

This module provides a robust interface for making requests to the OpenAI API
with exponential backoff retry logic to handle transient failures. It includes
comprehensive error handling for common API exceptions.

Dependencies:
    - tenacity: For retry logic
    - openai: For API interaction
    - python-dotenv: For environment variable management

Author: Brandon Jackson
Version: 1.0.1
"""

import logging
import os

from dotenv import load_dotenv
from openai import (APIError, AuthenticationError, BadRequestError, OpenAI,
                    OpenAIError, RateLimitError)
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((APIError, RateLimitError))
)
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
        AuthenticationError: If API key is invalid or missing
        BadRequestError: If the request is malformed or invalid
        RateLimitError: If rate limit is exceeded (will retry)
        APIError: If API encounters an error (will retry)
        ValueError: If the message dictionary is missing required keys
        Exception: For unexpected errors

    Example:
        >>> message = {"role": "user", "content": "List ten holiday destinations."}
        >>> response = get_response("gpt-4", message)
        >>> print(response)
    """
    try:
        # Validate input parameters
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Message must be a dictionary with 'role' and 'content' keys")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")

        # Make API request
        response = client.chat.completions.create(
            model=model,
            messages=[message]
        )
        
        return response.choices[0].message.content

    except AuthenticationError as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise

    except BadRequestError as e:
        logger.error(f"Bad request error: {str(e)}")
        raise

    except RateLimitError as e:
        # This will be caught by the retry decorator
        logger.warning(f"Rate limit exceeded: {str(e)}")
        raise

    except APIError as e:
        # This will be caught by the retry decorator
        logger.warning(f"API error occurred: {str(e)}")
        raise

    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        test_message = {"role": "user", "content": "List ten holiday destinations."}
        response = get_response("gpt-4", test_message)
        print(response)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")