from utils.nlp_utils import OpenAINLPEngine
from typing import Dict, List, Any

class SemanticImprovementAdvisor:
    def __init__(self, nlp_engine):
        self.nlp_engine = nlp_engine
    
    def generate_recommendations(self, content: str, analysis_results: Dict, target_keywords: List[str]) -> List[Dict]:
        """Generate recommendations to improve semantic relevance for target keywords."""
        try:
            entities = analysis_results.get("entities", [])
            topics = analysis_results.get("topics", [])
            headings = analysis_results.get("headings", {})
            
            recommendations = self.nlp_engine.generate_improvement_recommendations(
                content, entities, topics, target_keywords, headings
            )
            
            return recommendations
        except Exception as e:
            raise Exception(f"Error generating improvement recommendations: {str(e)}")
