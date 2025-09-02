import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_utils import OpenAINLPEngine
from utils.error_utils import NLPProcessingError, APIKeyError

class TestOpenAINLPEngine(unittest.TestCase):
    """Test cases for OpenAI NLP Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock API key for testing
        self.api_key = "test_api_key"
        
        # Sample text for testing
        self.sample_text = (
            "Google has announced a new update to its search algorithm that will focus on semantic "
            "understanding. The update, called MUM (Multitask Unified Model), aims to better understand "
            "the context and meaning behind search queries. This follows their previous BERT update from 2019. "
            "SEO experts are recommending that website owners focus on creating comprehensive content "
            "that addresses user intent rather than just including keywords."
        )
    
    @patch('openai.chat.completions.create')
    def test_extract_entities(self, mock_create):
        """Test entity extraction functionality."""
        # Mock OpenAI API response for entity extraction
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.arguments = json.dumps({
            "entities": [
                {
                    "name": "Google",
                    "type": "ORGANIZATION",
                    "prominence": 0.9,
                    "key_mentions": ["Google has announced"]
                },
                {
                    "name": "MUM",
                    "type": "PRODUCT",
                    "prominence": 0.8,
                    "key_mentions": ["called MUM (Multitask Unified Model)"]
                },
                {
                    "name": "BERT",
                    "type": "PRODUCT",
                    "prominence": 0.7,
                    "key_mentions": ["previous BERT update"]
                }
            ]
        })
        mock_create.return_value = mock_response
        
        # Initialize engine and extract entities
        engine = OpenAINLPEngine(api_key=self.api_key)
        entities = engine.extract_entities(self.sample_text)
        
        # Assertions
        self.assertEqual(len(entities), 3)
        self.assertEqual(entities[0]["name"], "Google")
        self.assertEqual(entities[0]["type"], "ORGANIZATION")
        self.assertEqual(entities[1]["name"], "MUM")
        self.assertEqual(entities[2]["name"], "BERT")
    
    @patch('openai.chat.completions.create')
    def test_identify_topics(self, mock_create):
        """Test topic identification functionality."""
        # Mock OpenAI API response for topic identification
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.arguments = json.dumps({
            "topics": [
                {
                    "name": "Search Algorithm Updates",
                    "category_path": "/Computers & Electronics/Search Engines",
                    "confidence": 0.9,
                    "is_primary": True
                },
                {
                    "name": "Semantic Search",
                    "category_path": "/Computers & Electronics/Search Engines/Natural Language Processing",
                    "confidence": 0.85,
                    "is_primary": False
                },
                {
                    "name": "SEO Best Practices",
                    "category_path": "/Marketing/Digital Marketing/SEO",
                    "confidence": 0.7,
                    "is_primary": False
                }
            ]
        })
        mock_create.return_value = mock_response
        
        # Initialize engine and identify topics
        engine = OpenAINLPEngine(api_key=self.api_key)
        topics = engine.identify_topics(self.sample_text)
        
        # Assertions
        self.assertEqual(len(topics), 3)
        self.assertEqual(topics[0]["name"], "Search Algorithm Updates")
        self.assertTrue(topics[0]["is_primary"])
        self.assertEqual(topics[1]["name"], "Semantic Search")
        self.assertEqual(topics[2]["name"], "SEO Best Practices")
    
    @patch('openai.chat.completions.create')
    def test_compute_semantic_similarity(self, mock_create):
        """Test semantic similarity computation."""
        # Mock OpenAI API response for semantic similarity
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.arguments = json.dumps({
            "scores": [
                {
                    "keyword": "semantic search",
                    "score": 85,
                    "justification": "The text directly discusses semantic understanding in search algorithms."
                },
                {
                    "keyword": "Google algorithm",
                    "score": 90,
                    "justification": "The text specifically mentions Google's search algorithm updates."
                },
                {
                    "keyword": "content marketing",
                    "score": 60,
                    "justification": "The text touches on content creation but doesn't focus on marketing aspects."
                }
            ],
            "average_score": 78.3
        })
        mock_create.return_value = mock_response
        
        # Initialize engine and compute similarity
        engine = OpenAINLPEngine(api_key=self.api_key)
        keywords = ["semantic search", "Google algorithm", "content marketing"]
        similarity = engine.compute_semantic_similarity(self.sample_text, keywords)
        
        # Assertions
        self.assertEqual(len(similarity["scores"]), 3)
        self.assertAlmostEqual(similarity["average_score"], 78.3, places=1)
        self.assertEqual(similarity["scores"][0]["keyword"], "semantic search")
        self.assertEqual(similarity["scores"][0]["score"], 85)
    
    @patch('openai.chat.completions.create')
    def test_api_error_handling(self, mock_create):
        """Test error handling for API issues."""
        # Mock API error
        mock_create.side_effect = Exception("API connection error")
        
        # Initialize engine
        engine = OpenAINLPEngine(api_key=self.api_key)
        
        # Test error handling for entity extraction
        entities = engine.extract_entities(self.sample_text)
        self.assertEqual(entities, [])
        
        # Test error handling for topic identification
        topics = engine.identify_topics(self.sample_text)
        self.assertEqual(topics, [])
        
        # Test error handling for semantic similarity
        similarity = engine.compute_semantic_similarity(self.sample_text, ["test"])
        self.assertEqual(similarity["scores"], [])
        self.assertEqual(similarity["average_score"], 0)
    
    def test_missing_api_key(self):
        """Test initialization with missing API key."""
        with self.assertRaises(Exception):
            OpenAINLPEngine(api_key=None)

if __name__ == "__main__":
    unittest.main()
