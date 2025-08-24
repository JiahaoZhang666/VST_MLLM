"""
Narrative Flow Predictor Module for VideoSpatioTemporal Framework

This module models causal relationships between events and predicts narrative flow,
enabling comprehensive understanding of storylines and causal coherence in videos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from .constants import (
    MAX_EVENTS, CAUSAL_TEMPORAL_THRESHOLD, NARRATIVE_COHERENCE_THRESHOLD,
    STORYLINE_MAX_LENGTH, EVENT_CATEGORIES
)


@dataclass
class Event:
    """Represents an event in the narrative flow."""
    event_id: int
    frame_id: int
    event_type: str
    entities_involved: List[int]
    description: str
    features: torch.Tensor
    confidence: float
    temporal_position: float  # Normalized position in video (0-1)


@dataclass
class CausalRelationship:
    """Represents a causal relationship between events."""
    source_event_id: int
    target_event_id: int
    causal_strength: float
    relationship_type: str
    temporal_gap: int
    features: torch.Tensor


@dataclass
class Storyline:
    """Represents a coherent storyline extracted from events."""
    storyline_id: int
    events: List[Event]
    causal_chain: List[CausalRelationship]
    coherence_score: float
    start_frame: int
    end_frame: int
    summary: str


class NarrativeFlowPredictor(nn.Module):
    """
    Predicts narrative flow and causal relationships in videos.
    
    Features:
    - Event detection and classification
    - Causal inference modeling
    - Storyline extraction
    - Narrative coherence assessment
    """
    
    def __init__(self, 
                 feature_dim: int = 512,
                 max_events: int = MAX_EVENTS,
                 max_storylines: int = 10,
                 hidden_dim: int = 256,
                 num_event_types: int = len(EVENT_CATEGORIES)):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_events = max_events
        self.max_storylines = max_storylines
        self.hidden_dim = hidden_dim
        self.num_event_types = num_event_types
        
        # Event detection network
        self.event_detector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_event_types + 1)  # +1 for background
        )
        
        # Event feature encoder
        self.event_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Causal relationship predictor
        self.causal_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, hidden_dim),  # +2 for temporal features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Narrative coherence assessor
        self.coherence_assessor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Storyline generator
        self.storyline_generator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Temporal modeling
        self.temporal_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Event memory
        self.event_memory = EventMemory(max_events=max_events)
        self.causal_memory = CausalMemory(max_events=max_events)
        
    def forward(self, 
                video_features: torch.Tensor,
                spatiotemporal_graph: Dict,
                audio_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Predict narrative flow from video features and spatiotemporal graph.
        
        Args:
            video_features: Video frame features
            spatiotemporal_graph: Spatiotemporal graph from graph builder
            audio_features: Optional audio features
            
        Returns:
            Dictionary containing:
            - events: Detected events
            - causal_relationships: Predicted causal relationships
            - storylines: Extracted storylines
            - narrative_coherence: Overall narrative coherence score
        """
        # Detect events
        events = self._detect_events(video_features, spatiotemporal_graph)
        
        # Encode event features
        event_features = self._encode_events(events)
        
        # Predict causal relationships
        causal_relationships = self._predict_causal_relationships(events, event_features)
        
        # Extract storylines
        storylines = self._extract_storylines(events, causal_relationships, event_features)
        
        # Assess narrative coherence
        narrative_coherence = self._assess_narrative_coherence(storylines, event_features)
        
        # Update memory
        self.event_memory.update(events)
        self.causal_memory.update(causal_relationships)
        
        return {
            'events': events,
            'causal_relationships': causal_relationships,
            'storylines': storylines,
            'narrative_coherence': narrative_coherence,
            'event_features': event_features
        }
    
    def _detect_events(self, 
                       video_features: torch.Tensor,
                       spatiotemporal_graph: Dict) -> List[Event]:
        """Detect events in the video based on spatiotemporal patterns."""
        events = []
        next_event_id = 0
        
        # Extract temporal features
        temporal_features, _ = self.temporal_encoder(video_features)
        
        # Process each frame for event detection
        for frame_idx in range(video_features.shape[0]):
            frame_features = temporal_features[frame_idx]
            
            # Get entities in current frame
            frame_entities = [n for n in spatiotemporal_graph['entities'] 
                            if n.frame_id == frame_idx]
            
            if not frame_entities:
                continue
            
            # Aggregate entity features for frame
            entity_features = torch.stack([e.features for e in frame_entities])
            frame_context = torch.mean(entity_features, dim=0)
            
            # Detect events
            event_logits = self.event_detector(frame_context)
            event_probs = F.softmax(event_logits, dim=-1)
            
            # Find significant events
            for event_type_idx, prob in enumerate(event_probs):
                if prob > 0.3 and event_type_idx < len(EVENT_CATEGORIES):  # Threshold
                    event_type = EVENT_CATEGORIES[event_type_idx]
                    
                    # Create event
                    event = Event(
                        event_id=next_event_id,
                        frame_id=frame_idx,
                        event_type=event_type,
                        entities_involved=[e.entity_id for e in frame_entities],
                        description=f"{event_type} at frame {frame_idx}",
                        features=frame_context,
                        confidence=prob.item(),
                        temporal_position=frame_idx / video_features.shape[0]
                    )
                    
                    events.append(event)
                    next_event_id += 1
        
        return events
    
    def _encode_events(self, events: List[Event]) -> torch.Tensor:
        """Encode event features for further processing."""
        if not events:
            return torch.empty(0, self.hidden_dim, device=next(iter(events)).features.device)
        
        event_features = torch.stack([e.features for e in events])
        encoded_features = self.event_encoder(event_features)
        
        return encoded_features
    
    def _predict_causal_relationships(self, 
                                    events: List[Event],
                                    event_features: torch.Tensor) -> List[CausalRelationship]:
        """Predict causal relationships between events."""
        causal_relationships = []
        
        if len(events) < 2:
            return causal_relationships
        
        # Consider all event pairs
        for i, event_i in enumerate(events):
            for j, event_j in enumerate(events):
                if i == j:
                    continue
                
                # Temporal constraint: cause must precede effect
                if event_i.frame_id >= event_j.frame_id:
                    continue
                
                # Calculate temporal gap
                temporal_gap = event_j.frame_id - event_i.frame_id
                
                # Predict causal strength
                causal_features = torch.cat([
                    event_features[i],
                    event_features[j],
                    torch.tensor([temporal_gap, event_j.temporal_position - event_i.temporal_position],
                               device=event_features.device)
                ])
                
                causal_strength = self.causal_predictor(causal_features)
                
                # Only keep strong causal relationships
                if causal_strength > CAUSAL_TEMPORAL_THRESHOLD:
                    # Determine relationship type
                    if temporal_gap <= 5:
                        relationship_type = "immediate_cause"
                    elif temporal_gap <= 20:
                        relationship_type = "short_term_cause"
                    else:
                        relationship_type = "long_term_cause"
                    
                    causal_rel = CausalRelationship(
                        source_event_id=event_i.event_id,
                        target_event_id=event_j.event_id,
                        causal_strength=causal_strength.item(),
                        relationship_type=relationship_type,
                        temporal_gap=temporal_gap,
                        features=causal_features
                    )
                    
                    causal_relationships.append(causal_rel)
        
        return causal_relationships
    
    def _extract_storylines(self, 
                           events: List[Event],
                           causal_relationships: List[CausalRelationship],
                           event_features: torch.Tensor) -> List[Storyline]:
        """Extract coherent storylines from events and causal relationships."""
        storylines = []
        next_storyline_id = 0
        
        if not events or not causal_relationships:
            return storylines
        
        # Build causal graph
        causal_graph = self._build_causal_graph(events, causal_relationships)
        
        # Find causal chains
        causal_chains = self._find_causal_chains(causal_graph, events)
        
        # Group chains into storylines
        for chain in causal_chains:
            if len(chain) >= 2:  # Minimum chain length
                # Calculate coherence
                coherence_score = self._calculate_chain_coherence(chain, event_features)
                
                if coherence_score > NARRATIVE_COHERENCE_THRESHOLD:
                    # Create storyline
                    storyline = Storyline(
                        storyline_id=next_storyline_id,
                        events=chain,
                        causal_chain=self._get_chain_causal_relationships(chain, causal_relationships),
                        coherence_score=coherence_score,
                        start_frame=min(e.frame_id for e in chain),
                        end_frame=max(e.frame_id for e in chain),
                        summary=self._generate_storyline_summary(chain)
                    )
                    
                    storylines.append(storyline)
                    next_storyline_id += 1
        
        # Limit number of storylines
        if len(storylines) > self.max_storylines:
            storylines = sorted(storylines, key=lambda s: s.coherence_score, reverse=True)
            storylines = storylines[:self.max_storylines]
        
        return storylines
    
    def _build_causal_graph(self, 
                           events: List[Event],
                           causal_relationships: List[CausalRelationship]) -> Dict[int, List[int]]:
        """Build a graph representation of causal relationships."""
        causal_graph = {event.event_id: [] for event in events}
        
        for rel in causal_relationships:
            if rel.source_event_id in causal_graph:
                causal_graph[rel.source_event_id].append(rel.target_event_id)
        
        return causal_graph
    
    def _find_causal_chains(self, 
                           causal_graph: Dict[int, List[int]],
                           events: List[Event]) -> List[List[Event]]:
        """Find causal chains in the causal graph."""
        chains = []
        visited = set()
        
        # Sort events by frame_id for temporal ordering
        sorted_events = sorted(events, key=lambda e: e.frame_id)
        
        for event in sorted_events:
            if event.event_id in visited:
                continue
            
            # Start new chain
            chain = [event]
            visited.add(event.event_id)
            
            # Extend chain
            self._extend_causal_chain(event.event_id, causal_graph, events, chain, visited)
            
            if len(chain) > 1:  # Only keep meaningful chains
                chains.append(chain)
        
        return chains
    
    def _extend_causal_chain(self, 
                            current_event_id: int,
                            causal_graph: Dict[int, List[int]],
                            events: List[Event],
                            chain: List[Event],
                            visited: set):
        """Recursively extend a causal chain."""
        if current_event_id not in causal_graph:
            return
        
        for next_event_id in causal_graph[current_event_id]:
            if next_event_id in visited:
                continue
            
            # Find event object
            next_event = next(e for e in events if e.event_id == next_event_id)
            
            # Add to chain
            chain.append(next_event)
            visited.add(next_event_id)
            
            # Continue extending
            self._extend_causal_chain(next_event_id, causal_graph, events, chain, visited)
    
    def _calculate_chain_coherence(self, 
                                  chain: List[Event],
                                  event_features: torch.Tensor) -> float:
        """Calculate the coherence score for a causal chain."""
        if len(chain) < 2:
            return 0.0
        
        # Get features for events in chain
        chain_indices = [i for i, e in enumerate(chain)]
        chain_features = event_features[chain_indices]
        
        # Calculate temporal consistency
        temporal_consistency = 1.0
        for i in range(1, len(chain)):
            if chain[i].frame_id <= chain[i-1].frame_id:
                temporal_consistency *= 0.5
        
        # Calculate feature consistency
        feature_consistency = torch.mean(torch.cosine_similarity(
            chain_features[:-1], chain_features[1:], dim=1
        )).item()
        
        # Combine scores
        coherence_score = (temporal_consistency + feature_consistency) / 2
        
        return coherence_score
    
    def _get_chain_causal_relationships(self, 
                                      chain: List[Event],
                                      causal_relationships: List[CausalRelationship]) -> List[CausalRelationship]:
        """Get causal relationships that belong to a specific chain."""
        chain_event_ids = {e.event_id for e in chain}
        
        chain_causal_rels = []
        for rel in causal_relationships:
            if (rel.source_event_id in chain_event_ids and 
                rel.target_event_id in chain_event_ids):
                chain_causal_rels.append(rel)
        
        return chain_causal_rels
    
    def _generate_storyline_summary(self, chain: List[Event]) -> str:
        """Generate a summary description of a storyline."""
        if not chain:
            return "Empty storyline"
        
        # Simple summary based on event types
        event_types = [e.event_type for e in chain]
        start_frame = chain[0].frame_id
        end_frame = chain[-1].frame_id
        
        summary = f"Storyline from frame {start_frame} to {end_frame}: "
        summary += " → ".join(event_types)
        
        return summary
    
    def _assess_narrative_coherence(self, 
                                   storylines: List[Storyline],
                                   event_features: torch.Tensor) -> float:
        """Assess overall narrative coherence of the video."""
        if not storylines:
            return 0.0
        
        # Calculate average storyline coherence
        avg_coherence = np.mean([s.coherence_score for s in storylines])
        
        # Calculate storyline diversity
        if len(storylines) > 1:
            storyline_features = []
            for storyline in storylines:
                if storyline.events:
                    storyline_feat = torch.mean(torch.stack([e.features for e in storyline.events]), dim=0)
                    storyline_features.append(storyline_feat)
            
            if storyline_features:
                storyline_features = torch.stack(storyline_features)
                diversity = torch.mean(torch.cosine_similarity(
                    storyline_features.unsqueeze(0), storyline_features.unsqueeze(1)
                )).item()
            else:
                diversity = 0.0
        else:
            diversity = 1.0
        
        # Combine coherence and diversity
        narrative_coherence = (avg_coherence + diversity) / 2
        
        return narrative_coherence
    
    def predict_future_events(self, 
                            current_events: List[Event],
                            causal_relationships: List[CausalRelationship],
                            prediction_horizon: int = 10) -> List[Dict]:
        """Predict future events based on current causal patterns."""
        predictions = []
        
        if not current_events or not causal_relationships:
            return predictions
        
        # Find recent high-confidence events
        recent_events = [e for e in current_events 
                        if e.confidence > 0.7 and e.frame_id >= max(e.frame_id for e in current_events) - 5]
        
        for event in recent_events:
            # Find causal relationships where this event is a cause
            causal_effects = [rel for rel in causal_relationships 
                            if rel.source_event_id == event.event_id]
            
            if causal_effects:
                # Predict future events based on causal patterns
                for effect_rel in causal_effects:
                    if effect_rel.temporal_gap <= prediction_horizon:
                        prediction = {
                            'source_event': event,
                            'predicted_event_type': 'causal_continuation',
                            'confidence': effect_rel.causal_strength,
                            'expected_frame': event.frame_id + effect_rel.temporal_gap,
                            'causal_strength': effect_rel.causal_strength
                        }
                        predictions.append(prediction)
        
        return predictions


class EventMemory(nn.Module):
    """Maintains memory of detected events for temporal consistency."""
    
    def __init__(self, max_events: int = MAX_EVENTS):
        super().__init__()
        self.max_events = max_events
        self.events = []
        self.event_counter = 0
    
    def update(self, new_events: List[Event]):
        """Update event memory with new events."""
        self.events.extend(new_events)
        
        # Maintain memory size
        if len(self.events) > self.max_events:
            # Remove oldest events
            self.events = sorted(self.events, key=lambda e: e.frame_id)
            self.events = self.events[-self.max_events:]
    
    def get_recent_events(self, window_size: int = 10) -> List[Event]:
        """Get events from recent frames."""
        if not self.events:
            return []
        
        recent_frame = max(e.frame_id for e in self.events)
        threshold = recent_frame - window_size
        
        return [e for e in self.events if e.frame_id >= threshold]


class CausalMemory(nn.Module):
    """Maintains memory of causal relationships for pattern recognition."""
    
    def __init__(self, max_events: int = MAX_EVENTS):
        super().__init__()
        self.max_events = max_events
        self.causal_patterns = []
    
    def update(self, new_relationships: List[CausalRelationship]):
        """Update causal memory with new relationships."""
        self.causal_patterns.extend(new_relationships)
        
        # Maintain memory size
        if len(self.causal_patterns) > self.max_events:
            # Remove oldest relationships
            self.causal_patterns = sorted(self.causal_patterns, 
                                        key=lambda r: r.source_event_id)
            self.causal_patterns = self.causal_patterns[-self.max_events:]
    
    def find_similar_patterns(self, 
                            target_relationship: CausalRelationship,
                            similarity_threshold: float = 0.8) -> List[CausalRelationship]:
        """Find similar causal patterns in memory."""
        similar_patterns = []
        
        for pattern in self.causal_patterns:
            # Calculate similarity based on relationship type and temporal gap
            if (pattern.relationship_type == target_relationship.relationship_type and
                abs(pattern.temporal_gap - target_relationship.temporal_gap) <= 2):
                
                # Feature similarity
                feature_similarity = F.cosine_similarity(
                    pattern.features.unsqueeze(0),
                    target_relationship.features.unsqueeze(0)
                ).item()
                
                if feature_similarity > similarity_threshold:
                    similar_patterns.append(pattern)
        
        return similar_patterns
