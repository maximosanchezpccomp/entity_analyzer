from utils.nlp_utils import OpenAINLPEngine
from typing import Dict, List, Any

class KeywordSemanticScorer:
    def __init__(self, nlp_engine):
        self.nlp_engine = nlp_engine
    
    def compute_similarity(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """Compute semantic similarity between content and keywords."""
        try:
            similarity_results = self.nlp_engine.compute_semantic_similarity(content, keywords)
            return similarity_results
        except Exception as e:
            raise Exception(f"Error computing keyword similarity: {str(e)}")
