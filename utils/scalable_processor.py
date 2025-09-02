"""Module for scalable content processing.

This module provides advanced functionality for processing large datasets
and scaling the application to handle multiple URLs and extended content.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from utils.url_utils import validate_url, extract_main_content
from utils.content_processor import split_content_for_analysis
from utils.error_utils import URLFetchError, ContentExtractionError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScalableContentProcessor:
    """Processor for handling content at scale."""
    
    def __init__(self, max_concurrent: int = 5, timeout: int = 30):
        """
        Initialize the scalable content processor.
        
        Args:
            max_concurrent: Maximum number of concurrent tasks
            timeout: Timeout for HTTP requests in seconds
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Set up async context manager."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context manager."""
        if self.session:
            await self.session.close()
    
    async def fetch_url_async(self, url: str) -> str:
        """
        Fetch URL content asynchronously.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content
        """
        if not validate_url(url):
            raise URLFetchError(url, "Invalid URL format")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            raise URLFetchError(url, f"Error fetching URL: {str(e)}")
        except asyncio.TimeoutError:
            raise URLFetchError(url, "Request timed out")
        except Exception as e:
            raise URLFetchError(url, f"Unexpected error: {str(e)}")
    
    async def process_url_batch(self, urls: List[str]) -> Dict[str, str]:
        """
        Process a batch of URLs asynchronously.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dictionary mapping URLs to their HTML content
        """
        tasks = []
        for url in urls:
            tasks.append(self.fetch_url_async(url))
        
        results = {}
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                html = await task
                results[urls[i]] = html
            except URLFetchError as e:
                logger.error(f"Error fetching {urls[i]}: {e.message}")
                results[urls[i]] = None
        
        return results
    
    async def process_url_list(self, urls: List[str], batch_size: int = 10) -> Dict[str, str]:
        """
        Process a list of URLs in batches.
        
        Args:
            urls: List of URLs to process
            batch_size: Size of each batch
            
        Returns:
            Dictionary mapping URLs to their HTML content
        """
        results = {}
        
        # Process URLs in batches
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            batch_results = await self.process_url_batch(batch)
            results.update(batch_results)
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(urls):
                await asyncio.sleep(1)
        
        return results
    
    def extract_content_batch(self, html_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Extract main content from a batch of HTML documents.
        
        Args:
            html_dict: Dictionary mapping URLs to HTML content
            
        Returns:
            Dictionary mapping URLs to extracted main content
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Create a map of futures to URLs
            future_to_url = {
                executor.submit(extract_main_content, html): url
                for url, html in html_dict.items()
                if html is not None
            }
            
            # Process results as they complete
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    content = future.result()
                    results[url] = content
                except ContentExtractionError as e:
                    logger.error(f"Error extracting content from {url}: {str(e)}")
                    results[url] = None
                except Exception as e:
                    logger.error(f"Unexpected error processing {url}: {str(e)}")
                    results[url] = None
        
        return results
    
    def chunk_content_for_processing(self, content: str, max_chunk_size: int = 4000) -> List[str]:
        """
        Split content into chunks suitable for NLP processing.
        
        Args:
            content: Text content to process
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of content chunks
        """
        return split_content_for_analysis(content, max_chunk_size)
    
    async def process_multiple_urls(self, urls: List[str]) -> Dict[str, str]:
        """
        Process multiple URLs from fetching to content extraction.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dictionary mapping URLs to extracted content
        """
        # Fetch HTML content for all URLs
        html_dict = await self.process_url_list(urls)
        
        # Extract main content from HTML
        content_dict = self.extract_content_batch(html_dict)
        
        return content_dict
    
    def create_dataframe_from_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert processing results to a DataFrame for analysis.
        
        Args:
            results: Dictionary of processing results
            
        Returns:
            DataFrame with results
        """
        data = []
        
        for url, content in results.items():
            row = {
                'url': url,
                'content': content,
                'content_length': len(content) if content else 0,
                'status': 'Success' if content else 'Failed',
                'timestamp': time.time()
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def process_results_in_batches(self, df: pd.DataFrame, batch_size: int = 10, 
                                  processor_func: callable = None) -> pd.DataFrame:
        """
        Process DataFrame results in batches using a provided function.
        
        Args:
            df: DataFrame with results
            batch_size: Size of each batch
            processor_func: Function to apply to each batch
            
        Returns:
            Processed DataFrame
        """
        if processor_func is None:
            return df
        
        processed_dfs = []
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].copy()
            processed_batch = processor_func(batch)
            processed_dfs.append(processed_batch)
            
        # Combine batches
        if processed_dfs:
            return pd.concat(processed_dfs, ignore_index=True)
        return df


# Utility functions for parallel processing

async def process_urls_concurrently(urls: List[str], max_concurrent: int = 5) -> Dict[str, str]:
    """
    Process a list of URLs concurrently.
    
    Args:
        urls: List of URLs to process
        max_concurrent: Maximum number of concurrent tasks
        
    Returns:
        Dictionary mapping URLs to their extracted content
    """
    async with ScalableContentProcessor(max_concurrent=max_concurrent) as processor:
        return await processor.process_multiple_urls(urls)

def run_async_processor(urls: List[str], max_concurrent: int = 5) -> Dict[str, str]:
    """
    Run the async processor from synchronous code.
    
    Args:
        urls: List of URLs to process
        max_concurrent: Maximum number of concurrent tasks
        
    Returns:
        Dictionary mapping URLs to their extracted content
    """
    return asyncio.run(process_urls_concurrently(urls, max_concurrent))
