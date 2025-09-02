"""Utility package for Semantic SEO Analyzer.

This package contains utility modules that provide functionality for:
1. URL processing and content extraction
2. NLP processing with OpenAI
3. Content processing and cleaning
4. Error handling
5. Data visualization
6. API interaction and caching
"""

# Import primary utility functions and classes for convenience
from utils.url_utils import validate_url, fetch_url_content, extract_main_content, extract_headings, get_url_metadata
from utils.nlp_utils import OpenAINLPEngine
from utils.content_processor import clean_text, split_content_for_analysis, extract_content_for_semantic_analysis
from utils.error_utils import SEOAnalyzerError, URLFetchError, ContentExtractionError, NLPProcessingError, APIKeyError
from utils.visualization import (
    create_entity_chart, 
    create_topic_chart, 
    create_keyword_similarity_chart, 
    create_recommendations_chart,
    create_eeat_radar_chart,
    create_semantic_overview_dashboard
)
from utils.openai_handler import OpenAIHandler

# Export the main functions and classes
__all__ = [
    # URL utilities
    'validate_url', 'fetch_url_content', 'extract_main_content', 'extract_headings', 'get_url_metadata',
    
    # NLP utilities
    'OpenAINLPEngine',
    
    # Content processing
    'clean_text', 'split_content_for_analysis', 'extract_content_for_semantic_analysis',
    
    # Error handling
    'SEOAnalyzerError', 'URLFetchError', 'ContentExtractionError', 'NLPProcessingError', 'APIKeyError',
    
    # Visualization
    'create_entity_chart', 'create_topic_chart', 'create_keyword_similarity_chart',
    'create_recommendations_chart', 'create_eeat_radar_chart', 'create_semantic_overview_dashboard',
    
    # API handling
    'OpenAIHandler'
]
