from utils.url_utils import fetch_url_content, extract_main_content, extract_headings, get_url_metadata
from utils.nlp_utils import OpenAINLPEngine
from typing import Dict, Any

class URLSemanticAnalyzer:
    def __init__(self, nlp_engine):
        self.nlp_engine = nlp_engine
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze the semantic content of a URL."""
        try:
            # Fetch and process URL content
            html_content = fetch_url_content(url)
            main_text = extract_main_content(html_content)
            headings = extract_headings(html_content)
            metadata = get_url_metadata(html_content, url)
            
            # Extract entities and topics
            entities = self.nlp_engine.extract_entities(main_text)
            topics = self.nlp_engine.identify_topics(main_text)
            
            # Infer Google's interpretation
            google_interpretation = self.nlp_engine.infer_google_interpretation(
                entities, topics, headings
            )
            
            # Return analysis results
            return {
                "url": url,
                "metadata": metadata,
                "entities": entities,
                "topics": topics,
                "headings": headings,
                "google_interpretation": google_interpretation
            }
            
        except Exception as e:
            raise Exception(f"Error during URL analysis: {str(e)}")
