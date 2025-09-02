import requests
from bs4 import BeautifulSoup
import trafilatura
import re
from urllib.parse import urlparse

def validate_url(url):
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def fetch_url_content(url):
    """Fetch HTML content from URL with error handling."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching URL content: {str(e)}")

def extract_main_content(html):
    """Extract main textual content from HTML using trafilatura."""
    try:
        # Try trafilatura for main content extraction
        main_text = trafilatura.extract(html)
        
        # If trafilatura fails, fall back to BeautifulSoup
        if not main_text:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
                
            main_text = soup.get_text(separator=' ')
            # Normalize whitespace
            main_text = re.sub(r'\s+', ' ', main_text).strip()
        
        return main_text
    except Exception as e:
        raise Exception(f"Error extracting main content: {str(e)}")

def extract_headings(html):
    """Extract headings (H1-H6) from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    headings = {}
    
    for i in range(1, 7):
        heading_tag = f'h{i}'
        headings[heading_tag] = [h.get_text().strip() for h in soup.find_all(heading_tag)]
    
    return headings

def get_url_metadata(html, url):
    """Extract metadata like title, description from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    metadata = {
        'title': None,
        'description': None,
        'canonical': url,
    }
    
    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text().strip()
    
    # Extract meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        metadata['description'] = meta_desc.get('content', '')
    
    # Extract canonical URL
    canonical = soup.find('link', attrs={'rel': 'canonical'})
    if canonical and canonical.get('href'):
        metadata['canonical'] = canonical.get('href')
    
    return metadata
