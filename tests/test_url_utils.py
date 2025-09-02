import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.url_utils import validate_url
from utils.error_utils import URLFetchError, ContentExtractionError

class TestURLUtils(unittest.TestCase):
    """Test cases for URL utilities."""
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        valid_urls = [
            "https://example.com",
            "https://www.example.com/page",
            "http://example.com/path/to/page.html",
            "https://subdomain.example.com/page?param=value",
            "https://example.com/page#section"
        ]
        
        for url in valid_urls:
            self.assertTrue(validate_url(url), f"URL should be valid: {url}")
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "",
            "example.com",  # Missing scheme
            "http://",      # Missing domain
            "ftp://example.com",  # Unsupported scheme
            "https:/example.com",  # Malformed URL
            "file:///path/to/file",  # File URL
            "javascript:alert('test')"  # JavaScript URL
        ]
        
        for url in invalid_urls:
            self.assertFalse(validate_url(url), f"URL should be invalid: {url}")
    
    def test_fetch_url_error_handling(self):
        """Test error handling during URL fetching."""
        # This test depends on external resources and network connection
        # Consider mocking these requests in a real testing environment
        pass
    
    def test_extract_content_error_handling(self):
        """Test error handling during content extraction."""
        # This test depends on external resources
        # Consider mocking these operations in a real testing environment
        pass

if __name__ == "__main__":
    unittest.main()
