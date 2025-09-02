"""Module for spaCy-based NLP engine.

This module provides a local NLP engine based on spaCy that can be used
as an alternative to the OpenAI API for offline processing or to reduce costs.
"""

import spacy
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpacyNLPEngine:
    """NLP engine based on spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize spaCy NLP engine.
        
        Args:
            model_name: Name of spaCy model to use (requires installation)
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{model_name}': {str(e)}")
            logger.error("Try installing the model with: python -m spacy download en_core_web_lg")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entity dictionaries
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            entity_texts = {}  # Track entities to avoid duplicates
            
            for ent in doc.ents:
                # Skip if already processed this entity text
                if ent.text in entity_texts:
                    continue
                
                # Map spaCy entity type to our standardized type
                entity_type = self._map_entity_type(ent.label_)
                
                # Calculate prominence based on frequency and position
                prominence = self._calculate_entity_prominence(ent, doc)
                
                # Get key mentions (contexts around entity)
                key_mentions = self._get_entity_mentions(text, ent.text, max_mentions=2)
                
                # Add to results
                entities.append({
                    "name": ent.text,
                    "type": entity_type,
                    "prominence": prominence,
                    "key_mentions": key_mentions
                })
                
                # Track processed entity text
                entity_texts[ent.text] = True
            
            # Sort by prominence
            entities = sorted(entities, key=lambda x: x["prominence"], reverse=True)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _map_entity_type(self, spacy_type: str) -> str:
        """Map spaCy entity types to standardized types."""
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "PRODUCT": "PRODUCT",
            "WORK_OF_ART": "WORK_OF_ART",
            "EVENT": "EVENT",
            "DATE": "DATE",
            "TIME": "DATE",
            "MONEY": "OTHER",
            "PERCENT": "OTHER",
            "FAC": "LOCATION",
            "NORP": "ORGANIZATION"  # Nationalities, religious or political groups
        }
        
        return mapping.get(spacy_type, "OTHER")
    
    def _calculate_entity_prominence(self, entity, doc) -> float:
        """
        Calculate entity prominence score.
        
        Args:
            entity: spaCy entity
            doc: spaCy document
            
        Returns:
            Prominence score (0-1)
        """
        # Count occurrences
        entity_text = entity.text.lower()
        text_lower = doc.text.lower()
        count = text_lower.count(entity_text)
        
        # Normalize by document length
        frequency = count / len(doc)
        
        # Adjust by position (earlier is more important)
        position_factor = 1.0 - (entity.start / len(doc))
        
        # Consider entity length (longer entities tend to be more specific)
        length_factor = min(len(entity_text) / 20, 1.0)  # Cap at 1.0
        
        # Combine factors (weighted average)
        prominence = (0.5 * frequency + 0.3 * position_factor + 0.2 * length_factor)
        
        # Ensure it's in 0-1 range
        return min(max(prominence * 10, 0.0), 1.0)  # Scale up and clamp
    
    def _get_entity_mentions(self, text: str, entity_text: str, max_mentions: int = 2) -> List[str]:
        """
        Get context snippets around entity mentions.
        
        Args:
            text: Full text
            entity_text: Entity text to find
            max_mentions: Maximum number of mentions to return
            
        Returns:
            List of context snippets
        """
        mentions = []
        
        # Use regex to find mentions (case insensitive)
        entity_pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
        matches = list(entity_pattern.finditer(text))
        
        # Get context for each match
        for i, match in enumerate(matches):
            if i >= max_mentions:
                break
                
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            
            # Extract context
            context = text[start:end]
            
            # Clean up whitespace
            context = re.sub(r'\s+', ' ', context).strip()
            
            mentions.append(context)
        
        return mentions
    
    def identify_topics(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify main and secondary topics in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of topic dictionaries
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract noun phrases as potential topics
            noun_chunks = list(doc.noun_chunks)
            
            # Count words and phrases
            word_counts = Counter()
            for token in doc:
                if not token.is_stop and not token.is_punct and token.is_alpha:
                    word_counts[token.lemma_] += 1
            
            # Count noun phrases
            phrase_counts = Counter()
            for chunk in noun_chunks:
                if len(chunk) > 1:  # Skip single-word chunks
                    phrase_counts[chunk.text] += 1
            
            # Combine counts for better topic identification
            combined_topics = {}
            
            # Process top phrases
            for phrase, count in phrase_counts.most_common(20):
                # Skip very common or very rare phrases
                if count < 2:
                    continue
                
                # Calculate confidence based on frequency and length
                frequency = count / len(noun_chunks) if noun_chunks else 0
                length_factor = min(len(phrase.split()) / 5, 1.0)  # Favor multi-word phrases
                confidence = min(frequency * 10 * length_factor, 1.0)
                
                # Skip low confidence topics
                if confidence < 0.3:
                    continue
                
                # Get category path using word vectors
                category_path = self._get_category_path(phrase, doc)
                
                # Add to combined topics
                combined_topics[phrase] = {
                    "name": phrase,
                    "category_path": category_path,
                    "confidence": confidence,
                    "is_primary": False  # Will set primary later
                }
            
            # Process top words
            for word, count in word_counts.most_common(30):
                if word in combined_topics or count < 3:
                    continue
                
                # Calculate confidence
                frequency = count / len(doc)
                confidence = min(frequency * 5, 0.9)  # Cap single words lower than phrases
                
                # Skip low confidence topics
                if confidence < 0.4:
                    continue
                
                # Get category path
                category_path = self._get_category_path(word, doc)
                
                # Add to combined topics
                combined_topics[word] = {
                    "name": word,
                    "category_path": category_path,
                    "confidence": confidence,
                    "is_primary": False
                }
            
            # Convert to list and sort by confidence
            topics = list(combined_topics.values())
            topics = sorted(topics, key=lambda x: x["confidence"], reverse=True)
            
            # Mark top topic as primary
            if topics:
                topics[0]["is_primary"] = True
            
            # Limit to top topics
            return topics[:10]
            
        except Exception as e:
            logger.error(f"Error identifying topics: {str(e)}")
            return []
    
    def _get_category_path(self, text: str, doc) -> str:
        """
        Determine category path for a topic.
        
        Args:
            text: Topic text
            doc: spaCy document
            
        Returns:
            Category path string
        """
        # Define major category vectors (simplified)
        categories = {
            "Technology": self.nlp("computer software hardware technology programming internet digital").vector,
            "Business": self.nlp("business company finance economy market investment").vector,
            "Health": self.nlp("health medical medicine disease doctor hospital").vector,
            "Science": self.nlp("science research scientific study experiment physics chemistry biology").vector,
            "Entertainment": self.nlp("entertainment movie film music show television").vector,
            "Sports": self.nlp("sports team player game competition tournament").vector,
            "Education": self.nlp("education school university student learn teaching").vector,
            "Politics": self.nlp("politics government policy law election political").vector,
            "Travel": self.nlp("travel tourism destination vacation holiday").vector
        }
        
        # Define subcategories for each major category
        subcategories = {
            "Technology": {
                "Software": self.nlp("software application program code development app").vector,
                "Hardware": self.nlp("hardware device computer smartphone gadget").vector,
                "Internet": self.nlp("internet website online digital web network").vector,
                "AI": self.nlp("artificial intelligence machine learning neural network AI algorithm").vector
            },
            "Business": {
                "Marketing": self.nlp("marketing advertising promotion brand campaign").vector,
                "Finance": self.nlp("finance investment banking money market stock").vector,
                "Entrepreneurship": self.nlp("startup entrepreneur business founder venture").vector
            },
            # More subcategories can be added as needed
        }
        
        # Get vector for the topic
        topic_vector = self.nlp(text).vector
        
        # Find best matching category
        best_category = "Other"
        best_similarity = -1
        
        for category, vector in categories.items():
            similarity = cosine_similarity([topic_vector], [vector])[0][0]
            if similarity > best_similarity and similarity > 0.3:
                best_similarity = similarity
                best_category = category
        
        # If we have subcategories for this category, find best match
        if best_category in subcategories:
            best_subcategory = None
            best_sub_similarity = -1
            
            for subcategory, vector in subcategories[best_category].items():
                similarity = cosine_similarity([topic_vector], [vector])[0][0]
                if similarity > best_sub_similarity and similarity > 0.3:
                    best_sub_similarity = similarity
                    best_subcategory = subcategory
            
            # Construct path with subcategory if found
            if best_subcategory:
                return f"/{best_category}/{best_subcategory}"
        
        return f"/{best_category}"
    
    def compute_semantic_similarity(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Compute semantic similarity between content and keywords.
        
        Args:
            content: Content text
            keywords: List of keywords
            
        Returns:
            Dictionary with similarity scores
        """
        try:
            if not content or not keywords:
                return {"scores": [], "average_score": 0}
            
            # Process content
            content_doc = self.nlp(content)
            
            # Calculate scores for each keyword
            scores = []
            total_score = 0
            
            for keyword in keywords:
                # Process keyword
                keyword_doc = self.nlp(keyword)
                
                # Calculate similarity using spaCy's word vectors
                similarity = content_doc.similarity(keyword_doc)
                
                # Convert to percentage (0-100 scale)
                score = round(similarity * 100)
                
                # Generate justification
                justification = self._generate_similarity_justification(
                    similarity, content_doc, keyword_doc
                )
                
                # Add to results
                scores.append({
                    "keyword": keyword,
                    "score": score,
                    "justification": justification
                })
                
                total_score += score
            
            # Calculate average
            average_score = total_score / len(keywords) if keywords else 0
            
            return {
                "scores": scores,
                "average_score": average_score
            }
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return {"scores": [], "average_score": 0}
    
    def _generate_similarity_justification(self, similarity: float, 
                                          content_doc, keyword_doc) -> str:
        """
        Generate a justification for the similarity score.
        
        Args:
            similarity: Similarity score
            content_doc: spaCy document for content
            keyword_doc: spaCy document for keyword
            
        Returns:
            Justification text
        """
        if similarity > 0.8:
            return "The content directly addresses this topic with substantial coverage."
        elif similarity > 0.6:
            return "The content contains several relevant references to this topic."
        elif similarity > 0.4:
            return "The content mentions some related concepts but lacks comprehensive coverage."
        elif similarity > 0.2:
            return "The content has minimal relevance to this topic."
        else:
            return "The content does not appear to address this topic."
    
    def generate_improvement_recommendations(self, content: str, entities: List[Dict], 
                                            topics: List[Dict], target_keywords: List[str],
                                            headings: Dict) -> List[Dict]:
        """
        Generate recommendations to improve semantic relevance.
        
        Args:
            content: Content text
            entities: Extracted entities
            topics: Identified topics
            target_keywords: Target keywords for optimization
            headings: Content headings structure
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = []
            
            # Process content and target keywords
            content_doc = self.nlp(content)
            keyword_docs = [self.nlp(kw) for kw in target_keywords]
            
            # 1. Check keyword presence in headings
            if headings and 'h1' in headings and headings['h1']:
                h1_text = headings['h1'][0] if isinstance(headings['h1'], list) else headings['h1']
                h1_doc = self.nlp(h1_text)
                
                keyword_in_h1 = False
                for kw_doc in keyword_docs:
                    if h1_doc.similarity(kw_doc) > 0.6:
                        keyword_in_h1 = True
                        break
                
                if not keyword_in_h1:
                    recommendations.append({
                        "priority": "High",
                        "category": "Content Structure",
                        "recommendation": f"Include target keywords in main heading (H1): '{h1_text}'",
                        "justification": "The main heading doesn't clearly indicate the target topic to search engines."
                    })
            
            # 2. Check for missing relevant entities
            # Generate potential relevant entities for target keywords
            relevant_entities = self._get_relevant_entities_for_keywords(target_keywords)
            
            # Check which ones are missing
            existing_entity_names = {e["name"].lower() for e in entities}
            for entity in relevant_entities:
                if entity["name"].lower() not in existing_entity_names:
                    recommendations.append({
                        "priority": "Medium",
                        "category": "Entity Improvement",
                        "recommendation": f"Add references to '{entity['name']}' which is semantically related to your target keywords",
                        "justification": f"This entity is strongly associated with the topic but is missing from your content."
                    })
            
            # 3. Check topic coverage
            topic_recommendations = self._analyze_topic_coverage(topics, keyword_docs)
            recommendations.extend(topic_recommendations)
            
            # 4. Check for missing schema based on content type
            schema_recommendation = self._suggest_schema_markup(content, entities, topics)
            if schema_recommendation:
                recommendations.append(schema_recommendation)
            
            # 5. Check for internal linking opportunities
            link_recommendation = {
                "priority": "Low",
                "category": "Internal Linking",
                "recommendation": "Create internal links to related content using descriptive anchor text that includes semantic variations of your target keywords",
                "justification": "Internal links with relevant anchor text help establish semantic relationships between pages."
            }
            recommendations.append(link_recommendation)
            
            # Sort by priority
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            recommendations.sort(key=lambda x: priority_order[x["priority"]])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _get_relevant_entities_for_keywords(self, keywords: List[str]) -> List[Dict]:
        """Generate relevant entities for target keywords."""
        relevant_entities = []
        
        # This would ideally use more sophisticated methods like knowledge graphs
        # For now, use simplified approach based on word vectors
        
        # Some predefined entity associations (simplified)
        entity_associations = {
            "seo": ["Google", "BERT", "Schema.org", "Search Console", "XML Sitemaps"],
            "content marketing": ["Content Strategy", "Buyer Persona", "Editorial Calendar", "ROI"],
            "social media": ["Facebook", "Instagram", "Twitter", "LinkedIn", "Engagement Rate"],
            "programming": ["Python", "JavaScript", "Git", "API", "Framework"],
            "technology": ["Machine Learning", "Blockchain", "Cloud Computing", "IoT"],
        }
        
        # Check keywords against known associations
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Check direct matches in our dictionary
            for topic, entities in entity_associations.items():
                similarity = self.nlp(keyword_lower).similarity(self.nlp(topic))
                
                if similarity > 0.6:
                    for entity in entities:
                        relevant_entities.append({
                            "name": entity,
                            "relevance": similarity
                        })
            
            # For keywords without direct matches, generate generic entities
            # (In a real implementation, this would use a knowledge graph or larger dataset)
            if not any(e for e in relevant_entities if e["relevance"] > 0.6):
                # Add some generic entities that are likely relevant
                relevant_entities.append({
                    "name": "Industry Experts",
                    "relevance": 0.7
                })
                relevant_entities.append({
                    "name": "Case Studies",
                    "relevance": 0.7
                })
        
        # Remove duplicates and sort by relevance
        unique_entities = {}
        for entity in relevant_entities:
            name = entity["name"]
            if name not in unique_entities or entity["relevance"] > unique_entities[name]["relevance"]:
                unique_entities[name] = entity
        
        return sorted(unique_entities.values(), key=lambda x: x["relevance"], reverse=True)
    
    def _analyze_topic_coverage(self, topics: List[Dict], keyword_docs) -> List[Dict]:
        """Analyze topic coverage compared to target keywords."""
        recommendations = []
        
        # Check if we have enough topics with good confidence
        if len(topics) < 3:
            recommendations.append({
                "priority": "High",
                "category": "Topic Coverage",
                "recommendation": "Expand content to cover more aspects of the main topic",
                "justification": "Content lacks sufficient topic depth for strong semantic relevance."
            })
        
        # Check if topics align with target keywords
        aligned_topics = 0
        for topic in topics:
            topic_doc = self.nlp(topic["name"])
            for kw_doc in keyword_docs:
                if topic_doc.similarity(kw_doc) > 0.6:
                    aligned_topics += 1
                    break
        
        if aligned_topics < min(2, len(keyword_docs)):
            recommendations.append({
                "priority": "Medium",
                "category": "Topic Coverage",
                "recommendation": "Better align content topics with target keywords",
                "justification": "There's insufficient semantic alignment between content topics and target keywords."
            })
        
        return recommendations
    
    def _suggest_schema_markup(self, content: str, entities: List[Dict], topics: List[Dict]) -> Optional[Dict]:
        """Suggest schema markup based on content analysis."""
        # Simplified schema suggestion based on content indicators
        # In a real implementation, this would be more sophisticated
        
        # Check for article indicators
        article_indicators = ["wrote", "published", "author", "article", "post", "blog"]
        is_article = any(indicator in content.lower() for indicator in article_indicators)
        
        # Check for product indicators
        product_indicators = ["price", "product", "buy", "purchase", "shipping", "stock"]
        is_product = any(indicator in content.lower() for indicator in product_indicators)
        
        # Check for FAQ indicators
        faq_indicators = ["question", "answer", "faq", "frequently asked", "q:", "q&a"]
        is_faq = any(indicator in content.lower() for indicator in faq_indicators)
        
        # Make schema recommendation
        if is_article:
            return {
                "priority": "Medium",
                "category": "Schema Markup",
                "recommendation": "Implement Article schema markup to define your content type for search engines",
                "justification": "Article schema helps search engines understand content type, author, and publish date."
            }
        elif is_product:
            return {
                "priority": "High",
                "category": "Schema Markup",
                "recommendation": "Implement Product schema markup with price, availability, and reviews",
                "justification": "Product schema can generate rich snippets in search results, increasing visibility."
            }
        elif is_faq:
            return {
                "priority": "Medium",
                "category": "Schema Markup",
                "recommendation": "Implement FAQPage schema markup for your questions and answers",
                "justification": "FAQ schema can generate rich snippets in search results, increasing SERP real estate."
            }
        else:
            return {
                "priority": "Low",
                "category": "Schema Markup",
                "recommendation": "Consider implementing WebPage or CreativeWork schema as a baseline",
                "justification": "Basic schema provides context about your content to search engines."
            }
    
    def infer_google_interpretation(self, entities: List[Dict], topics: List[Dict], 
                                  headings: Dict) -> Dict[str, Any]:
        """
        Infer how Google might interpret the content.
        
        Args:
            entities: Extracted entities
            topics: Identified topics
            headings: Content headings
            
        Returns:
            Dictionary with interpretation data
        """
        try:
            # Determine main topic
            main_topic = topics[0]["name"] if topics else "Unknown"
            
            # Get secondary topics (excluding main)
            secondary_topics = [t["name"] for t in topics[1:5] if t["name"] != main_topic]
            
            # Get key entities
            key_entities = [e["name"] for e in entities[:5]] if entities else []
            
            # Infer E-E-A-T assessment
            eeat_assessment = self._assess_eeat(entities, topics, headings)
            
            # Identify strengths
            strengths = []
            
            # Check for topic clarity
            if topics and topics[0]["confidence"] > 0.7:
                strengths.append("Clear topical focus that search engines can easily identify")
            
            # Check for heading structure
            if headings and 'h1' in headings and 'h2' in headings:
                if headings['h1'] and len(headings.get('h2', [])) >= 2:
                    strengths.append("Well-structured content with logical heading hierarchy")
            
            # Check for entity richness
            if len(entities) > 10:
                strengths.append("Rich entity presence helps establish topical relevance")
            
            # Identify weaknesses
            weaknesses = []
            
            # Check for topic confusion
            if not topics or (topics and topics[0]["confidence"] < 0.5):
                weaknesses.append("Unclear main topic may confuse search algorithms")
            
            # Check for heading structure issues
            if not headings or 'h1' not in headings or not headings.get('h1'):
                weaknesses.append("Missing H1 heading affects content structure signals")
            
            # Check for thin content indicators
            if len(entities) < 5:
                weaknesses.append("Limited entity presence suggests thin content")
            
            # Ensure we have some default values
            if not strengths:
                strengths.append("Basic content structure is present")
            
            if not weaknesses:
                weaknesses.append("No major semantic weaknesses detected")
            
            return {
                "main_topic": main_topic,
                "secondary_topics": secondary_topics,
                "key_entities": key_entities,
                "eeat_assessment": eeat_assessment,
                "strengths": strengths,
                "weaknesses": weaknesses
            }
            
        except Exception as e:
            logger.error(f"Error inferring Google interpretation: {str(e)}")
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
    
    def _assess_eeat(self, entities: List[Dict], topics: List[Dict], headings: Dict) -> Dict[str, str]:
        """Assess E-E-A-T signals in content."""
        # This is a simplified assessment - a real implementation would be more sophisticated
        
        # Check for expertise signals
        expertise_signals = {
            "expert_entities": 0,
            "technical_terms": 0,
            "detailed_explanations": False
        }
        
        # Count expert entities (people, organizations)
        for entity in entities:
            if entity["type"] in ["PERSON", "ORGANIZATION"]:
                expertise_signals["expert_entities"] += 1
        
        # Check for technical terms in topics
        technical_terms = ["algorithm", "research", "study", "analysis", "methodology", 
                          "data", "statistics", "framework", "theory", "concept"]
        
        for topic in topics:
            if any(term in topic["name"].lower() for term in technical_terms):
                expertise_signals["technical_terms"] += 1
        
        # Check for detailed explanations via headings
        if headings and 'h2' in headings and 'h3' in headings:
            if len(headings.get('h2', [])) >= 3 and len(headings.get('h3', [])) >= 3:
                expertise_signals["detailed_explanations"] = True
        
        # Assess expertise
        if expertise_signals["expert_entities"] >= 3 or expertise_signals["technical_terms"] >= 3 or expertise_signals["detailed_explanations"]:
            expertise = "Strong expertise signals with technical terminology and structured explanations"
        elif expertise_signals["expert_entities"] >= 1 or expertise_signals["technical_terms"] >= 1:
            expertise = "Moderate expertise signals present but could be strengthened"
        else:
            expertise = "Limited expertise signals detected"
        
        # Assess authoritativeness (simplified)
        if expertise_signals["expert_entities"] >= 2:
            authoritativeness = "References to authoritative sources strengthen credibility"
        else:
            authoritativeness = "Limited authority signals, consider adding references to recognized sources"
        
        # Assess trustworthiness (simplified)
        trust_entities = ["study", "research", "source", "evidence", "data", "statistics"]
        has_trust_signals = any(e["name"].lower() in trust_entities for e in entities)
        
        if has_trust_signals:
            trustworthiness = "Content includes trust signals like data references"
        else:
            trustworthiness = "Consider adding more evidence-based references to build trust"
        
        return {
            "expertise": expertise,
            "authoritativeness": authoritativeness,
            "trustworthiness": trustworthiness
        }
