"""
MindMapGPT - Dynamic Knowledge Visualization System
Core architecture for AI-powered mind mapping tool
"""

import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import re
from collections import defaultdict, Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

@dataclass
class MindMapNode:
    """Represents a single node in the mind map"""
    id: str
    text: str
    category: str
    importance_score: float
    position: Tuple[float, float] = (0, 0)
    color: str = "#4A90E2"
    size: int = 20

@dataclass
class MindMapEdge:
    """Represents a connection between nodes"""
    source: str
    target: str
    weight: float
    relationship_type: str = "related"

class TextProcessor:
    """Handles NLP processing and concept extraction"""
    
    def __init__(self):
        # Initialize spaCy model with fallback
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded successfully")
        except (ImportError, OSError) as e:
            print("⚠ Warning: spaCy not available. Using basic text processing.")
            print("  To install: pip install spacy && python -m spacy download en_core_web_sm")
        
        # Initialize NLTK with fallback
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
            print("✓ NLTK stopwords loaded successfully")
        except Exception as e:
            print("⚠ Warning: NLTK not fully available. Using basic stopwords.")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Initialize TF-IDF vectorizer
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),  # Reduced to improve performance
                min_df=1,  # Allow single occurrences
                max_df=0.95  # Remove very common terms
            )
            print("✓ TF-IDF vectorizer initialized successfully")
        except ImportError:
            print("⚠ Warning: scikit-learn not available. Install with: pip install scikit-learn")
            self.vectorizer = None
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def extract_key_concepts(self, text: str, num_concepts: int = 20) -> List[Tuple[str, float]]:
        """Extract key concepts from text using multiple methods"""
        if not text or not text.strip():
            return []
        
        concepts = []
        print(f"Processing text of length: {len(text)}")
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(text)
        sentences = self._simple_sentence_split(cleaned_text)
        print(f"Split into {len(sentences)} sentences")
        
        # Method 1: Basic keyword extraction (always available)
        basic_concepts = self._extract_basic_keywords(text, num_concepts // 2)
        concepts.extend(basic_concepts)
        print(f"Extracted {len(basic_concepts)} basic concepts")
        
        # Method 2: TF-IDF if available
        if self.vectorizer and len(sentences) > 1:
            try:
                tfidf_concepts = self._extract_tfidf_concepts(sentences, num_concepts // 2)
                concepts.extend(tfidf_concepts)
                print(f"Extracted {len(tfidf_concepts)} TF-IDF concepts")
            except Exception as e:
                print(f"TF-IDF extraction failed: {e}")
        
        # Method 3: Named entities if spaCy is available
        if self.nlp:
            try:
                ner_concepts = self._extract_named_entities(text)
                concepts.extend(ner_concepts)
                print(f"Extracted {len(ner_concepts)} named entities")
            except Exception as e:
                print(f"NER extraction failed: {e}")
        
        # Remove duplicates and sort by importance
        concept_dict = {}
        for concept, score in concepts:
            concept_clean = concept.strip().lower()
            if len(concept_clean) > 2 and concept_clean not in self.stop_words:
                if concept_clean in concept_dict:
                    concept_dict[concept_clean] = max(concept_dict[concept_clean], score)
                else:
                    concept_dict[concept_clean] = score
        
        result = sorted(concept_dict.items(), key=lambda x: x[1], reverse=True)[:num_concepts]
        print(f"Final concept count: {len(result)}")
        
        if not result:
            # Fallback: extract any meaningful words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = Counter(words)
            result = [(word, freq/len(words)) for word, freq in word_freq.most_common(min(num_concepts, 10))]
            print(f"Fallback extraction: {len(result)} concepts")
        
        return result
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting when NLTK is not available"""
        try:
            import nltk
            return nltk.sent_tokenize(text)
        except:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _extract_basic_keywords(self, text: str, num_concepts: int) -> List[Tuple[str, float]]:
        """Extract keywords using basic frequency analysis"""
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stopwords
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        total_words = len(filtered_words)
        
        if total_words == 0:
            return []
        
        # Convert to normalized scores
        concepts = [(word, freq/total_words) for word, freq in word_freq.most_common(num_concepts)]
        
        return concepts
    
    def _extract_tfidf_concepts(self, sentences: List[str], num_concepts: int) -> List[Tuple[str, float]]:
        """Extract concepts using TF-IDF"""
        if len(sentences) < 1 or not self.vectorizer:
            return []
        
        try:
            # Handle single sentence case
            if len(sentences) == 1:
                sentences = sentences + ['']  # Add empty sentence to make TF-IDF work
            
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores across all documents
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top scoring terms
            top_indices = np.argsort(mean_scores)[::-1][:num_concepts]
            
            result = [(feature_names[i], float(mean_scores[i])) for i in top_indices if mean_scores[i] > 0]
            return result
        except Exception as e:
            print(f"Error in TF-IDF extraction: {e}")
            return []
    
    def _extract_named_entities(self, text: str) -> List[Tuple[str, float]]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if len(ent.text) > 2 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                # Score based on frequency and entity type importance
                importance = 0.8 if ent.label_ in ['PERSON', 'ORG'] else 0.6
                entities.append((ent.text.lower(), importance))
        
        return entities
    
    def find_relationships(self, concepts: List[str], text: str) -> List[Tuple[str, str, float]]:
        """Find relationships between concepts using co-occurrence and semantic similarity"""
        if len(concepts) < 2:
            return []
        
        relationships = []
        sentences = self._simple_sentence_split(text.lower())
        
        print(f"Finding relationships between {len(concepts)} concepts in {len(sentences)} sentences")
        
        # Co-occurrence based relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                similarity = self._calculate_relationship_strength(concept1, concept2, sentences, text)
                if similarity > 0.05:  # Lower threshold for better connectivity
                    relationships.append((concept1, concept2, similarity))
        
        print(f"Found {len(relationships)} relationships")
        return relationships
    
    def _calculate_relationship_strength(self, concept1: str, concept2: str, sentences: List[str], full_text: str) -> float:
        """Calculate relationship strength between two concepts"""
        if not sentences:
            return 0.0
        
        # Method 1: Co-occurrence in same sentence
        cooccurrence_count = 0
        proximity_score = 0
        
        for sentence in sentences:
            if concept1 in sentence and concept2 in sentence:
                cooccurrence_count += 1
                
                # Calculate word distance within sentence
                words = sentence.split()
                try:
                    pos1 = next(i for i, word in enumerate(words) if concept1 in word)
                    pos2 = next(i for i, word in enumerate(words) if concept2 in word)
                    distance = abs(pos1 - pos2)
                    proximity_score += 1.0 / (distance + 1)  # Closer words = higher score
                except (StopIteration, ValueError):
                    continue
        
        # Method 2: Check in same paragraph
        paragraphs = full_text.split('\n')
        paragraph_cooccurrence = 0
        for paragraph in paragraphs:
            if concept1 in paragraph.lower() and concept2 in paragraph.lower():
                paragraph_cooccurrence += 1
        
        # Normalize scores
        total_sentences = len(sentences)
        total_paragraphs = len(paragraphs)
        
        sentence_score = cooccurrence_count / total_sentences if total_sentences > 0 else 0
        paragraph_score = paragraph_cooccurrence / total_paragraphs if total_paragraphs > 0 else 0
        proximity_normalized = proximity_score / max(1, cooccurrence_count) if cooccurrence_count > 0 else 0
        
        # Add semantic similarity
        semantic_score = self._simple_semantic_similarity(concept1, concept2)
        
        # Weighted combination
        final_score = (sentence_score * 0.4) + (paragraph_score * 0.2) + (proximity_normalized * 0.2) + (semantic_score * 0.2)
        
        return min(1.0, final_score)  # Cap at 1.0
    
    def _simple_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Simple semantic similarity based on word overlap"""
        words1 = set(concept1.split())
        words2 = set(concept2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class MindMapGenerator:
    """Generates mind map structure from processed concepts"""
    
    def __init__(self):
        self.processor = TextProcessor()
        self.graph = nx.Graph()
    
    def generate_mindmap(self, text: str, num_concepts: int = 15) -> Tuple[List[MindMapNode], List[MindMapEdge]]:
        """Generate complete mind map from input text"""
        # Extract key concepts
        concepts_with_scores = self.processor.extract_key_concepts(text, num_concepts)
        concepts = [concept for concept, _ in concepts_with_scores]
        
        # Find relationships
        relationships = self.processor.find_relationships(concepts, text)
        
        # Create nodes
        nodes = self._create_nodes(concepts_with_scores)
        
        # Create edges
        edges = self._create_edges(relationships)
        
        # Position nodes using force-directed layout
        self._position_nodes(nodes, edges)
        
        return nodes, edges
    
    def _create_nodes(self, concepts_with_scores: List[Tuple[str, float]]) -> List[MindMapNode]:
        """Create mind map nodes from concepts"""
        nodes = []
        categories = self._categorize_concepts([c[0] for c in concepts_with_scores])
        
        for i, (concept, score) in enumerate(concepts_with_scores):
            node = MindMapNode(
                id=f"node_{i}",
                text=concept.title(),
                category=categories.get(concept, "general"),
                importance_score=score,
                size=max(15, min(40, int(score * 50))),  # Scale size based on importance
                color=self._get_category_color(categories.get(concept, "general"))
            )
            nodes.append(node)
        
        return nodes
    
    def _create_edges(self, relationships: List[Tuple[str, str, float]]) -> List[MindMapEdge]:
        """Create mind map edges from relationships"""
        edges = []
        
        # Create concept to node ID mapping (simplified for this example)
        concept_to_id = {}
        node_counter = 0
        
        for source, target, weight in relationships:
            if source not in concept_to_id:
                concept_to_id[source] = f"node_{node_counter}"
                node_counter += 1
            if target not in concept_to_id:
                concept_to_id[target] = f"node_{node_counter}"
                node_counter += 1
            
            edge = MindMapEdge(
                source=concept_to_id[source],
                target=concept_to_id[target],
                weight=weight,
                relationship_type="semantic" if weight > 0.5 else "related"
            )
            edges.append(edge)
        
        return edges
    
    def _categorize_concepts(self, concepts: List[str]) -> Dict[str, str]:
        """Categorize concepts for better organization"""
        categories = {}
        
        # Simple keyword-based categorization (can be enhanced with ML)
        category_keywords = {
            "technology": ["software", "system", "data", "algorithm", "computer", "digital", "tech"],
            "business": ["company", "market", "customer", "revenue", "strategy", "management"],
            "science": ["research", "study", "analysis", "method", "theory", "experiment"],
            "people": ["team", "user", "person", "group", "community", "individual"],
            "process": ["process", "workflow", "procedure", "method", "approach", "implementation"]
        }
        
        for concept in concepts:
            concept_lower = concept.lower()
            assigned_category = "general"
            
            for category, keywords in category_keywords.items():
                if any(keyword in concept_lower for keyword in keywords):
                    assigned_category = category
                    break
            
            categories[concept] = assigned_category
        
        return categories
    
    def _get_category_color(self, category: str) -> str:
        """Get color for category"""
        color_map = {
            "technology": "#4A90E2",
            "business": "#F5A623",
            "science": "#7ED321",
            "people": "#D0021B",
            "process": "#9013FE",
            "general": "#50E3C2"
        }
        return color_map.get(category, "#50E3C2")
    
    def _position_nodes(self, nodes: List[MindMapNode], edges: List[MindMapEdge]):
        """Position nodes using NetworkX layout algorithms"""
        # Create NetworkX graph
        G = nx.Graph()
        
        for node in nodes:
            G.add_node(node.id, weight=node.importance_score)
        
        for edge in edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Use spring layout for positioning
        if len(G.nodes()) > 1:
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Update node positions
            for node in nodes:
                if node.id in pos:
                    node.position = (pos[node.id][0] * 400, pos[node.id][1] * 400)

class MindMapExporter:
    """Export mind maps to various formats"""
    
    @staticmethod
    def to_json(nodes: List[MindMapNode], edges: List[MindMapEdge]) -> str:
        """Export to JSON format"""
        data = {
            "nodes": [asdict(node) for node in nodes],
            "edges": [asdict(edge) for edge in edges]
        }
        return json.dumps(data, indent=2)
    
    @staticmethod
    def to_graphml(nodes: List[MindMapNode], edges: List[MindMapEdge]) -> str:
        """Export to GraphML format for use with graph visualization tools"""
        G = nx.Graph()
        
        for node in nodes:
            G.add_node(node.id, 
                      text=node.text,
                      category=node.category,
                      importance=node.importance_score,
                      color=node.color)
        
        for edge in edges:
            G.add_edge(edge.source, edge.target, 
                      weight=edge.weight,
                      relationship=edge.relationship_type)
        
        # Convert to GraphML string (simplified)
        return nx.generate_graphml(G)

# Example usage and testing
if __name__ == "__main__":
    # Sample text for testing
    sample_text = """
    Artificial intelligence and machine learning are transforming the technology industry.
    Companies are investing heavily in data science and algorithm development.
    The software engineering teams work closely with data scientists to implement
    these AI systems. Customer feedback and user experience research drive
    product development decisions. Market analysis shows growing demand for
    AI-powered solutions across various business sectors.
    """
    
    # Generate mind map
    generator = MindMapGenerator()
    nodes, edges = generator.generate_mindmap(sample_text)
    
    # Export to JSON
    exporter = MindMapExporter()
    json_output = exporter.to_json(nodes, edges)
    
    print("Generated Mind Map:")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print("\nSample nodes:")
    for node in nodes[:5]:
        print(f"- {node.text} (Category: {node.category}, Score: {node.importance_score:.3f})")
    
    print(f"\nJSON export preview (first 500 chars):")
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)