import openai
import json
from typing import List, Dict, Any

class OpenAINLPEngine:
    def __init__(self, api_key=None):
        """Initialize OpenAI NLP Engine with optional API key."""
        if api_key:
            openai.api_key = api_key
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using OpenAI API."""
        # Truncate text if too long (GPT models have context limits)
        max_length = 4000  # Adjust based on model limits
        truncated_text = text[:max_length] if len(text) > max_length else text
        
        try:
            # Function calling to get structured entity data
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",  # Using a model with function calling capability
                messages=[
                    {"role": "system", "content": "Extract named entities from the provided text. Identify entities such as PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, DATE, etc. For each entity, provide its name, type, and a prominence score from 0.0 to 1.0 based on its importance in the text."},
                    {"role": "user", "content": truncated_text}
                ],
                functions=[
                    {
                        "name": "extract_entities",
                        "description": "Extract named entities with their types and prominence scores",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "entities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string", "enum": ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "DATE", "WORK_OF_ART", "CONSUMER_GOOD", "OTHER"]},
                                            "prominence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                            "key_mentions": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["name", "type", "prominence"]
                                    }
                                }
                            },
                            "required": ["entities"]
                        }
                    }
                ],
                function_call={"name": "extract_entities"}
            )
            
            # Extract entities from function call
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return function_args.get("entities", [])
            
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return []
    
    def identify_topics(self, text: str) -> List[Dict[str, Any]]:
        """Identify main and secondary topics in the text."""
        max_length = 4000
        truncated_text = text[:max_length] if len(text) > max_length else text
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Identify the main and secondary topics in the text. For each topic, provide a category/subcategory path (like /Computers & Electronics/Software) and a confidence score from 0.0 to 1.0."},
                    {"role": "user", "content": truncated_text}
                ],
                functions=[
                    {
                        "name": "identify_topics",
                        "description": "Identify main and secondary topics with their hierarchical categories and confidence scores",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topics": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "category_path": {"type": "string"},
                                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                            "is_primary": {"type": "boolean"}
                                        },
                                        "required": ["name", "category_path", "confidence", "is_primary"]
                                    }
                                }
                            },
                            "required": ["topics"]
                        }
                    }
                ],
                function_call={"name": "identify_topics"}
            )
            
            # Extract topics from function call
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return function_args.get("topics", [])
            
        except Exception as e:
            print(f"Error identifying topics: {str(e)}")
            return []
    
    def compute_semantic_similarity(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Compute semantic similarity between content and keywords."""
        max_length = 4000
        truncated_content = content[:max_length] if len(content) > max_length else content
        
        try:
            # Create a prompt that asks for semantic similarity scores
            keywords_str = ", ".join(keywords)
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Analyze the semantic similarity between the provided content and keywords. For each keyword, provide a similarity score from 0 to 100 based on how semantically relevant the content is to that keyword."},
                    {"role": "user", "content": f"Content: {truncated_content}\n\nKeywords: {keywords_str}"}
                ],
                functions=[
                    {
                        "name": "similarity_scores",
                        "description": "Provide semantic similarity scores for each keyword",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "scores": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "keyword": {"type": "string"},
                                            "score": {"type": "number", "minimum": 0, "maximum": 100},
                                            "justification": {"type": "string"}
                                        },
                                        "required": ["keyword", "score"]
                                    }
                                },
                                "average_score": {"type": "number", "minimum": 0, "maximum": 100}
                            },
                            "required": ["scores", "average_score"]
                        }
                    }
                ],
                function_call={"name": "similarity_scores"}
            )
            
            # Extract similarity scores from function call
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            
            # Convert to dictionary format for easier access
            scores_dict = {
                "scores": function_args.get("scores", []),
                "average_score": function_args.get("average_score", 0)
            }
            
            return scores_dict
            
        except Exception as e:
            print(f"Error computing semantic similarity: {str(e)}")
            return {"scores": [], "average_score": 0}
    
    def generate_improvement_recommendations(self, content: str, entities: List[Dict], topics: List[Dict], 
                                            target_keywords: List[str], headings: Dict) -> List[Dict]:
        """Generate recommendations to improve semantic relevance for target keywords."""
        try:
            # Prepare input data
            entities_json = json.dumps(entities[:10])  # Limit to top 10 entities
            topics_json = json.dumps(topics)
            headings_json = json.dumps(headings)
            keywords_str = ", ".join(target_keywords)
            
            # Create a prompt for recommendations
            response = openai.chat.completions.create(
                model="gpt-4o",  # Using GPT-4 for more sophisticated recommendations
                messages=[
                    {"role": "system", "content": "Generate prioritized recommendations to improve the semantic relevance of a webpage for specific target keywords. Focus on entity gaps, topic coverage, content structure, and E-E-A-T signals."},
                    {"role": "user", "content": f"Content analysis results:\n- Target Keywords: {keywords_str}\n- Identified Entities: {entities_json}\n- Identified Topics: {topics_json}\n- Content Headings: {headings_json}\n\nBased on this analysis, provide prioritized recommendations to improve the semantic relevance of the content for the target keywords."}
                ],
                functions=[
                    {
                        "name": "improvement_recommendations",
                        "description": "Provide prioritized recommendations for semantic improvement",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "recommendations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                                            "category": {"type": "string", "enum": ["Entity Improvement", "Topic Coverage", "Content Structure", "Schema Markup", "Internal Linking", "E-E-A-T Signals"]},
                                            "recommendation": {"type": "string"},
                                            "justification": {"type": "string"}
                                        },
                                        "required": ["priority", "category", "recommendation", "justification"]
                                    }
                                }
                            },
                            "required": ["recommendations"]
                        }
                    }
                ],
                function_call={"name": "improvement_recommendations"}
            )
            
            # Extract recommendations from function call
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return function_args.get("recommendations", [])
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []

    def infer_google_interpretation(self, entities: List[Dict], topics: List[Dict], headings: Dict) -> Dict:
        """Infer how Google might interpret the content based on entities, topics, and structure."""
        try:
            # Prepare input data
            entities_json = json.dumps(entities[:10])  # Limit to top entities
            topics_json = json.dumps(topics)
            headings_json = json.dumps(headings)
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Analyze how Google might interpret a webpage based on its entities, topics, and structure. Consider E-E-A-T signals, BERT contextual understanding, and Knowledge Graph connections."},
                    {"role": "user", "content": f"Content analysis:\n- Entities: {entities_json}\n- Topics: {topics_json}\n- Content Structure: {headings_json}\n\nBased on this data, infer how Google's algorithms might interpret and evaluate this content."}
                ],
                functions=[
                    {
                        "name": "google_interpretation",
                        "description": "Infer Google's interpretation of the content",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "main_topic": {"type": "string"},
                                "secondary_topics": {"type": "array", "items": {"type": "string"}},
                                "key_entities": {"type": "array", "items": {"type": "string"}},
                                "eeat_assessment": {
                                    "type": "object",
                                    "properties": {
                                        "expertise": {"type": "string"},
                                        "authoritativeness": {"type": "string"},
                                        "trustworthiness": {"type": "string"}
                                    }
                                },
                                "strengths": {"type": "array", "items": {"type": "string"}},
                                "weaknesses": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["main_topic", "key_entities", "eeat_assessment", "strengths", "weaknesses"]
                        }
                    }
                ],
                function_call={"name": "google_interpretation"}
            )
            
            # Extract interpretation from function call
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return function_args
            
        except Exception as e:
            print(f"Error inferring Google interpretation: {str(e)}")
            return {
                "main_topic": "Unknown",
                "secondary_topics": [],
                "key_entities": [],
                "eeat_assessment": {
                    "expertise": "Unable to assess",
                    "authoritativeness": "Unable to assess",
                    "trustworthiness": "Unable to assess"
                },
                "strengths": [],
                "weaknesses": []
            }
