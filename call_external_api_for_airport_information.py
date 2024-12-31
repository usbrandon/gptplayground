import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
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

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

def get_airport_info(airport_code: str) -> Dict[str, Any]:
    """
    Call the Aviation API to get information about an airport.
    
    Args:
        airport_code (str): The IATA/ICAO code of the airport
        
    Returns:
        Dict[str, Any]: Airport information
        
    Raises:
        requests.RequestException: If the API request fails
    """
    api_key = os.getenv('AVIATION_API_KEY')
    if not api_key:
        raise ValueError("Aviation API key not found in environment variables")
        
    url = f"https://aviation-api.com/v1/airports/{airport_code}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching airport info: {str(e)}")
        raise

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
    Send a request to OpenAI API with support for aviation API function calling.
    
    Args:
        model (str): The OpenAI model identifier to use
        messages (list): List of message dictionaries
        tools (list, optional): List of function definitions
        return_function_call (bool): Whether to return function call arguments
        
    Returns:
        Union[str, dict]: Message content or function call arguments
    """
    try:
        # Input validation
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries")
            
        api_params = {
            "model": model,
            "messages": messages
        }
        
        if tools:
            api_params["tools"] = tools
            
        response = client.chat.completions.create(**api_params)
        
        if return_function_call and hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            return response.choices[0].message.tool_calls[0].function.arguments
            
        return response.choices[0].message.content

    except (AuthenticationError, BadRequestError) as e:
        logger.error(f"API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Define the aviation API function
        function_definitions = [{
            "type": "function",
            "function": {
                "name": "get_airport_info",
                "description": "Get information about an airport using its code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "airport_code": {
                            "type": "string",
                            "description": "The IATA/ICAO airport code"
                        }
                    },
                    "required": ["airport_code"]
                }
            }
        }]

        # Example messages for testing
        messages = [{
            "role": "user",
            "content": "What information can you tell me about JFK airport?"
        }]

        # Get response with function calling
        response = get_response_with_tools(
            "gpt-4",
            messages,
            tools=function_definitions,
            return_function_call=True
        )
        
        # If we get a function call response, execute it
        if isinstance(response, dict) and "airport_code" in response:
            airport_info = get_airport_info(response["airport_code"])
            print(f"Airport Information: {json.dumps(airport_info, indent=2)}")
        else:
            print(f"Response: {response}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")