import openai
import json
import time
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from config.api_config import APIConfig
from utils.error_utils import NLPProcessingError, APIKeyError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIHandler:
    """Enhanced handler for OpenAI API interactions with caching and retry logic."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[APIConfig] = None):
        """
        Initialize OpenAI API handler.
        
        Args:
            api_key: OpenAI API key
            config: Optional API configuration object
        """
        if not api_key:
            raise APIKeyError("OpenAI", "API key is required")
        
        self.api_key = api_key
        openai.api_key = api_key
        
        # Use default config if none provided
        self.config = config or APIConfig()
        
        # Initialize cache
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, model: str, messages: List[Dict], functions: Optional[List] = None) -> str:
        """Generate a unique cache key for the request."""
        # Create a dictionary of the key parameters
        key_data = {
            "model": model,
            "messages": messages,
            "functions": functions
        }
        
        # Convert to JSON and hash
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()
        
        return key_hash
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if it exists."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading from cache: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: Dict) -> None:
        """Cache response for future use."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def call_chat_completion(
        self, 
        messages: List[Dict],
        model: Optional[str] = None,
        functions: Optional[List] = None,
        function_call: Optional[Dict] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Call OpenAI chat completion API with retry logic and caching.
        
        Args:
            messages: List of message objects
            model: OpenAI model to use
            functions: Optional functions for function calling
            function_call: Optional function call specification
            temperature: Optional temperature parameter
            use_cache: Whether to use caching
            
        Returns:
            API response dictionary
        """
        # Get configuration values
        model = model or self.config.get("openai", "default_model")
        temperature = temperature or self.config.get("openai", "temperature")
        max_retries = self.config.get("openai", "retry_count")
        rate_limit_wait = self.config.get("openai", "rate_limit_wait")
        request_timeout = self.config.get("openai", "request_timeout")
        
        # Check if response is cached
        if use_cache:
            cache_key = self._generate_cache_key(model, messages, functions)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                logger.info(f"Using cached response for {model}")
                return cached_response
        
        # API request with retries
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Prepare request parameters
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": request_timeout
                }
                
                # Add function calling if provided
                if functions:
                    request_kwargs["functions"] = functions
                    
                    if function_call:
                        request_kwargs["function_call"] = function_call
                
                # Make API call
                response = openai.chat.completions.create(**request_kwargs)
                
                # Convert to dictionary for easier handling
                response_dict = {
                    "id": response.id,
                    "model": response.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content
                            },
                            "finish_reason": choice.finish_reason
                        }
                        for choice in response.choices
                    ]
                }
                
                # Add function call if present
                if hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
                    function_call = response.choices[0].message.function_call
                    response_dict["choices"][0]["message"]["function_call"] = {
                        "name": function_call.name,
                        "arguments": function_call.arguments
                    }
                
                # Cache the response
                if use_cache:
                    self._cache_response(cache_key, response_dict)
                
                return response_dict
                
            except openai.RateLimitError:
                logger.warning(f"Rate limit exceeded, waiting {rate_limit_wait} seconds...")
                time.sleep(rate_limit_wait)
                retry_count += 1
                
            except openai.APIError as e:
                logger.error(f"API error: {str(e)}")
                
                if "maximum context length" in str(e).lower():
                    raise NLPProcessingError("chat_completion", "Content exceeds maximum context length")
                    
                retry_count += 1
                time.sleep(2 ** retry_count)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise NLPProcessingError("chat_completion", f"Error during API call: {str(e)}")
        
        # If we've exhausted all retries
        raise NLPProcessingError("chat_completion", "Maximum retries exceeded")
    
    def extract_function_args(self, response: Dict) -> Dict:
        """
        Extract function call arguments from API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Function arguments as dictionary
        """
        try:
            # Get function call from first choice
            function_call = response.get("choices", [{}])[0].get("message", {}).get("function_call", {})
            
            if not function_call:
                return {}
            
            # Parse arguments JSON
            arguments = function_call.get("arguments", "{}")
            return json.loads(arguments)
            
        except Exception as e:
            logger.error(f"Error extracting function arguments: {str(e)}")
            return {}
    
    def extract_content(self, response: Dict) -> str:
        """
        Extract text content from API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Text content
        """
        try:
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return ""
    
    def process_content_in_chunks(
        self, 
        content_chunks: List[str],
        system_prompt: str,
        user_prompt_template: str,
        model: Optional[str] = None,
        functions: Optional[List] = None,
        function_call: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process content in chunks and combine results.
        
        Args:
            content_chunks: List of content chunks
            system_prompt: System prompt
            user_prompt_template: User prompt template with {content} placeholder
            model: OpenAI model to use
            functions: Optional functions for function calling
            function_call: Optional function call specification
            
        Returns:
            List of result dictionaries from each chunk
        """
        results = []
        
        for i, chunk in enumerate(content_chunks):
            logger.info(f"Processing chunk {i+1} of {len(content_chunks)}")
            
            # Create messages for this chunk
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(content=chunk)}
            ]
            
            # Call API
            response = self.call_chat_completion(
                messages=messages,
                model=model,
                functions=functions,
                function_call=function_call
            )
            
            # Extract and store function arguments
            if functions and function_call:
                args = self.extract_function_args(response)
                results.append(args)
            else:
                # Extract and store content
                content = self.extract_content(response)
                results.append({"content": content})
        
        return results
    
    def combine_chunk_results(self, results: List[Dict], result_key: str) -> List:
        """
        Combine results from multiple chunks.
        
        Args:
            results: List of result dictionaries
            result_key: Key to extract from each result
            
        Returns:
            Combined list of items
        """
        combined = []
        
        for result in results:
            items = result.get(result_key, [])
            if items:
                combined.extend(items)
        
        return combined
