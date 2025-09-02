import re
from bs4 import BeautifulSoup
from utils.error_utils import ContentExtractionError

def clean_text(text):
    """Clean extracted text by normalizing whitespace and fixing common issues."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common boilerplate phrases
    boilerplate = [
        "Accept all cookies",
        "We use cookies",
        "This website uses cookies",
        "Privacy Policy",
        "Terms of Service",
        "All Rights Reserved"
    ]
    
    for phrase in boilerplate:
        text = text.replace(phrase, "")
    
    # Normalize quotes and apostrophes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    return text.strip()

def extract_schema_markup(html):
    """Extract schema.org markup from the HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    schema_scripts = soup.find_all('script', type='application/ld+json')
    
    schemas = []
    for script in schema_scripts:
        try:
            schemas.append(script.string)
        except:
            pass
    
    return schemas

def split_content_for_analysis(text, max_length=4000):
    """Split content into chunks for API processing with context preservation."""
    if len(text) <= max_length:
        return [text]
    
    # Try to split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_length:
            current_chunk += para + "\n\n"
        else:
            # If adding this paragraph exceeds max_length, start a new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is longer than max_length, split it by sentences
            if len(para) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                inner_chunk = ""
                
                for sentence in sentences:
                    if len(inner_chunk) + len(sentence) + 1 <= max_length:
                        inner_chunk += sentence + " "
                    else:
                        if inner_chunk:
                            chunks.append(inner_chunk.strip())
                        
                        # If sentence itself is too long, split it by max_length
                        if len(sentence) > max_length:
                            for i in range(0, len(sentence), max_length - 100):
                                chunks.append(sentence[i:i + max_length - 100].strip())
                        else:
                            inner_chunk = sentence + " "
                
                if inner_chunk:
                    chunks.append(inner_chunk.strip())
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_internal_links(html, base_url):
    """Extract internal links from HTML."""
    internal_links = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        domain = re.search(r'https?://(?:www\.)?([^/]+)', base_url).group(1)
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '')
            
            # Skip empty, javascript, or anchor links
            if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                continue
                
            # Convert relative URLs to absolute
            if href.startswith('/'):
                if base_url.endswith('/'):
                    href = base_url + href[1:]
                else:
                    href = base_url + href
            
            # Check if link is internal
            if domain in href:
                link_text = a_tag.get_text().strip()
                if link_text:  # Only include links with text
                    internal_links.append({
                        'url': href,
                        'text': link_text[:100]  # Limit text length
                    })
    except Exception as e:
        raise ContentExtractionError(message=f"Error extracting internal links: {str(e)}")
    
    return internal_links

def extract_content_for_semantic_analysis(html):
    """
    Extract and structure content specifically for semantic analysis.
    Returns relevant parts separately for focused analysis.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract main content
        main_content = ""
        main_tags = soup.find_all(['article', 'main', 'div.content', 'div.main'])
        if main_tags:
            for tag in main_tags:
                content = tag.get_text()
                if len(content) > len(main_content):
                    main_content = content
        
        # If no main tags found, use bodytext as fallback
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text()
        
        # Clean the extracted content
        main_content = clean_text(main_content)
        
        # Extract headings with hierarchy
        headings = {}
        for i in range(1, 7):
            tag_name = f'h{i}'
            headings[tag_name] = [clean_text(h.get_text()) for h in soup.find_all(tag_name)]
        
        # Extract image alt texts
        image_alts = [img.get('alt', '') for img in soup.find_all('img') if img.get('alt')]
        
        # Extract meta information
        meta_info = {
            'title': clean_text(soup.title.string) if soup.title else '',
            'description': '',
            'keywords': ''
        }
        
        description_meta = soup.find('meta', attrs={'name': 'description'})
        if description_meta:
            meta_info['description'] = clean_text(description_meta.get('content', ''))
            
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta:
            meta_info['keywords'] = clean_text(keywords_meta.get('content', ''))
        
        return {
            'main_content': main_content,
            'headings': headings,
            'image_alts': image_alts,
            'meta_info': meta_info
        }
        
    except Exception as e:
        raise ContentExtractionError(message=f"Error extracting content for semantic analysis: {str(e)}")
