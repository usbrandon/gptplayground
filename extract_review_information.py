"""
OpenAI API interaction module with function calling, retry and error handling functionality.

This module provides a robust interface for making requests to the OpenAI API
with support for function calling and exponential backoff retry logic.

Dependencies:
    - tenacity: For retry logic
    - openai: For API interaction
    - python-dotenv: For environment variable management

Author: Brandon Jackson (Modified)
Version: 1.1.0
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tiktoken
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

def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the JSON file

    Returns:
        Dict[str, Any]: Parsed JSON content

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
            
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def load_function_definition(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a function definition from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the JSON file containing the function definition

    Returns:
        List[Dict[str, Any]]: Function definition in the format expected by the OpenAI API

    Example:
        >>> functions = load_function_definition('resources/real_estate_info_function.json')
        >>> print(functions[0]['function']['name'])
        'get_real_estate_info'
    """
    try:
        function_def = load_json_file(file_path)
        if isinstance(function_def, dict):
            # If the JSON contains a single function definition, wrap it in a list
            return [function_def]
        elif isinstance(function_def, list):
            return function_def
        else:
            raise ValueError("Function definition must be either a dictionary or a list of dictionaries")
    except Exception as e:
        logger.error(f"Error loading function definition: {str(e)}")
        raise



def num_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """Calculate the total number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((APIError, RateLimitError))
)
def get_response_with_tools(
    model: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    return_function_call: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Send a request to OpenAI API with support for function calling and get the response.

    Args:
        model (str): The OpenAI model identifier to use (e.g., "gpt-4", "gpt-3.5-turbo")
        messages (list): List of message dictionaries with 'role' and 'content' keys
        tools (list, optional): List of function definitions for the model to use
        return_function_call (bool): If True, returns the function call arguments instead of message content

    Returns:
        Union[str, dict]: Either the message content or function call arguments based on return_function_call

    Raises:
        AuthenticationError: If API key is invalid or missing
        BadRequestError: If the request is malformed or invalid
        RateLimitError: If rate limit is exceeded (will retry)
        APIError: If API encounters an error (will retry)
        ValueError: If the input parameters are invalid
        Exception: For unexpected errors

    Example:
        >>> messages = [{"role": "user", "content": "What's the weather in London?"}]
        >>> tools = [{
        ...     "type": "function",
        ...     "function": {
        ...         "name": "get_weather",
        ...         "description": "Get the weather in a location",
        ...         "parameters": {
        ...             "type": "object",
        ...             "properties": {
        ...                 "location": {"type": "string"},
        ...                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        ...             },
        ...             "required": ["location"]
        ...         }
        ...     }
        ... }]
        >>> response = get_response_with_tools("gpt-4", messages, tools, return_function_call=True)
    """
    try:
        # Validate input parameters
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries")
            
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys")
        
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")

        # Prepare API request parameters
        api_params = {
            "model": model,
            "messages": messages
        }

        # Add tools if provided
        if tools:
            if not isinstance(tools, list):
                raise ValueError("Tools must be a list of function definitions")
            api_params["tools"] = tools

        # Make API request
        response = client.chat.completions.create(**api_params)
        
        # Check if return_function_call is requested and tool_calls are available
        if return_function_call and hasattr(response.choices[0].message, 'tool_calls'):
            return [
                tool_call.function.arguments
                for choice in response.choices
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls
                for tool_call in choice.message.tool_calls
            ]

        # Otherwise, return all available message contents
        return [
            choice.message.content
            for choice in response.choices
            if hasattr(choice.message, 'content') and choice.message.content
        ]


    except AuthenticationError as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise

    except BadRequestError as e:
        logger.error(f"Bad request error: {str(e)}")
        raise

    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {str(e)}")
        raise

    except APIError as e:
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
        # Load function definition and prompt from files
        function_def = load_function_definition('./resources/extract_review_info_function.json')
        messages = load_function_definition('./resources/order_information_prompt.json')
        # Example function definition for weather
      
        # Check total tokens
        total_tokens = num_tokens_from_messages(messages, model="gpt-4o-mini")
        TOKEN_LIMIT = 8192  # Adjust based on model

        if total_tokens <= TOKEN_LIMIT:
            # Get response with function calling
            response = get_response_with_tools(
                "gpt-4o-mini",
                messages,
                tools=function_def,
                return_function_call=True
            )

            parsed_data = [json.loads(item) for item in response]
            print("Function call arguments:")
            print(json.dumps(parsed_data, indent=2))
        else:
            print(f"Message exceeds token limit. Total tokens: {total_tokens}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")