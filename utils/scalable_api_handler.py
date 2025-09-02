"""Module for scalable API handling.

This module provides functionality for handling API requests at scale,
with features like rate limiting, request batching, and load balancing
between multiple API keys.
"""

import asyncio
import time
import random
import json
import hashlib
import os
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import aiohttp
import openai
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests: int = 60, time_period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests in time period
            time_period: Time period in seconds
        """
        self.max_requests = max_requests
        self.time_period = time_period
        self.request_timestamps = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove timestamps older than the time period
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                    if now - ts <= self.time_period]
        
        # If at rate limit, wait until we can make another request
        if len(self.request_timestamps) >= self.max_requests:
            oldest = min(self.request_timestamps)
            wait_time = oldest + self.time_period - now
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(time.time())


class APIKeyManager:
    """Manager for multiple API keys with load balancing."""
    
    def __init__(self, api_keys: List[str], requests_per_minute: int = 60):
        """
        Initialize API key manager.
        
        Args:
            api_keys: List of API keys
            requests_per_minute: Rate limit per key (requests per minute)
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        self.api_keys = api_keys
        self.rate_limiters = {key: RateLimiter(requests_per_minute, 60) for key in api_keys}
        self.key_index = 0
    
    async def get_next_available_key(self) -> str:
        """
        Get the next available API key, respecting rate limits.
        
        Returns:
            Available API key
        """
        # Try all keys in rotation
        for _ in range(len(self.api_keys)):
            # Get next key in rotation
            key = self.api_keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            
            # Wait if this key is rate limited
            await self.rate_limiters[key].wait_if_needed()
            
            # Return the key
            return key
        
        # If we get here, all keys are rate limited
        # Just wait for the first key and return it
        key = self.api_keys[0]
        await self.rate_limiters[key].wait_if_needed()
        return key


class ScalableAPIHandler:
    """Scalable handler for API interactions."""
    
    def __init__(self, api_keys: Union[str, List[str]], cache_dir: Optional[str] = None,
                requests_per_minute: int = 60):
        """
        Initialize scalable API handler.
        
        Args:
            api_keys: Single API key or list of keys
            cache_dir: Directory for caching responses
            requests_per_minute: Rate limit per key (requests per minute)
        """
        # Convert single key to list
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        
        # Set up key manager
        self.key_manager = APIKeyManager(api_keys, requests_per_minute)
        
        # Set up cache
        self.use_cache = cache_dir is not None
        self.cache_dir = cache_dir
        if self.use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for request parameters.
        
        Args:
            params: Request parameters
            
        Returns:
            Cache key
        """
        # Create a string representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        
        # Generate MD5 hash
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached response or None
        """
        if not self.use_cache:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """
        Cache response.
        
        Args:
            cache_key: Cache key
            response: Response to cache
        """
        if not self.use_cache:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
    
    async def call_openai_chat_async(self, 
                              messages: List[Dict[str, str]],
                              model: str = "gpt-3.5-turbo-0125",
                              temperature: float = 0.2,
                              functions: Optional[List[Dict]] = None,
                              function_call: Optional[Dict] = None,
                              use_cache: bool = True) -> Dict[str, Any]:
        """
        Call OpenAI Chat Completion API asynchronously.
        
        Args:
            messages: List of message objects
            model: Model to use
            temperature: Temperature parameter
            functions: Functions for function calling
            function_call: Function call specification
            use_cache: Whether to use cache
            
        Returns:
            API response
        """
        # Prepare parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if functions:
            params["functions"] = functions
            
            if function_call:
                params["function_call"] = function_call
        
        # Check cache
        if use_cache and self.use_cache:
            cache_key = self._generate_cache_key(params)
            cached = self._get_cached_response(cache_key)
            
            if cached:
                logger.info(f"Using cached response for {model}")
                return cached
        
        # Get API key
        api_key = await self.key_manager.get_next_available_key()
        
        # Make API call
        try:
            # Configure client
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Make request
            response = await client.chat.completions.create(**params)
            
            # Convert to dict for caching and consistent handling
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
            
            # Cache response
            if use_cache and self.use_cache:
                self._cache_response(cache_key, response_dict)
            
            return response_dict
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    async def batch_process(self, 
                           items: List[Any],
                           process_func: Callable[[Any], Dict],
                           max_concurrency: int = 5,
                           **kwargs) -> List[Dict]:
        """
        Process a batch of items concurrently.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            max_concurrency: Maximum concurrent tasks
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processing results
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_limit(item):
            async with semaphore:
                return await process_func(item, **kwargs)
        
        # Create tasks
        tasks = [process_with_limit(item) for item in items]
        
        # Run tasks and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing item {i}: {str(result)}")
                processed_results.append({"error": str(result), "item": items[i]})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def analyze_large_content(self,
                                   content: str,
                                   chunk_size: int = 4000,
                                   system_prompt: str = "",
                                   user_prompt_template: str = "{content}",
                                   model: str = "gpt-3.5-turbo-0125",
                                   **kwargs) -> List[Dict]:
        """
        Analyze large content by splitting it into chunks.
        
        Args:
            content: Content to analyze
            chunk_size: Maximum chunk size
            system_prompt: System prompt
            user_prompt_template: User prompt template with {content} placeholder
            model: Model to use
            **kwargs: Additional parameters for API call
            
        Returns:
            List of analysis results for each chunk
        """
        from utils.content_processor import split_content_for_analysis
        
        # Split content into chunks
        chunks = split_content_for_analysis(content, chunk_size)
        logger.info(f"Split content into {len(chunks)} chunks")
        
        # Process function for each chunk
        async def process_chunk(chunk):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(content=chunk)}
            ]
            
            return await self.call_openai_chat_async(
                messages=messages,
                model=model,
                **kwargs
            )
        
        # Process all chunks
        return await self.batch_process(
            items=chunks,
            process_func=process_chunk,
            max_concurrency=3  # Lower concurrency for API calls
        )
    
    def extract_function_args_from_responses(self, responses: List[Dict], combine_arrays: bool = True) -> List[Dict]:
        """
        Extract function call arguments from a list of API responses.
        
        Args:
            responses: List of API responses
            combine_arrays: Whether to combine array fields across responses
            
        Returns:
            List of extracted function arguments
        """
        extracted = []
        combined = {}
        
        for response in responses:
            try:
                # Get function call from response
                function_call = response.get("choices", [{}])[0].get("message", {}).get("function_call", {})
                
                if not function_call:
                    continue
                
                # Parse arguments
                arguments = function_call.get("arguments", "{}")
                args = json.loads(arguments)
                
                if combine_arrays:
                    # Combine arrays across responses
                    for key, value in args.items():
                        if isinstance(value, list):
                            if key not in combined:
                                combined[key] = []
                            combined[key].extend(value)
                        else:
                            combined[key] = value
                else:
                    # Just collect individual results
                    extracted.append(args)
                
            except Exception as e:
                logger.error(f"Error extracting function arguments: {str(e)}")
        
        # Return combined result if combining, otherwise list of results
        if combine_arrays and combined:
            return [combined]
        return extracted


# Utility function for using the handler from synchronous code
def run_api_batch_process(items: List[Any], process_func: Callable, api_keys: Union[str, List[str]], 
                        max_concurrency: int = 5, **kwargs) -> List[Dict]:
    """
    Run batch processing from synchronous code.
    
    Args:
        items: List of items to process
        process_func: Async function to process each item
        api_keys: API key(s) to use
        max_concurrency: Maximum concurrent tasks
        **kwargs: Additional arguments for process_func
        
    Returns:
        List of processing results
    """
    async def run_batch():
        handler = ScalableAPIHandler(api_keys)
        return await handler.batch_process(items, process_func, max_concurrency, **kwargs)
    
    return asyncio.run(run_batch())
