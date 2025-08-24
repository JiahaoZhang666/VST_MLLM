"""
Task-Specific Output Heads Module for VideoSpatioTemporal Framework

This module implements specialized output heads for different tasks:
- VideoQA: Question answering about video content
- Relationship Recognition: Modeling long-term interactions between people
- Event Prediction: Predicting future actions or states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from .gtr_former import GTRFormer
from .actor_graph import ActorNode, ActorEdge, EdgeType


@dataclass
class VideoQAOutput:
    """Output for VideoQA task."""
    answer: str
    confidence: float
    reasoning_path: List[str]
    supporting_evidence: List[Dict]


@dataclass
class RelationshipOutput:
    """Output for relationship recognition task."""
    relationship_type: str
    confidence: float
    actor1_id: int
    actor2_id: int
    relationship_features: torch.Tensor
    interaction_pattern: List[Dict]


@dataclass
class EventPredictionOutput:
    """Output for event prediction task."""
    predicted_event: str
    confidence: float
    time_horizon: int
    affected_actors: List[int]
    causal_factors: List[str]


class VideoQAHead(nn.Module):
    """Video Question Answering head."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_answers: int = 1000,
                 max_question_length: int = 128):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_answers = num_answers
        self.max_question_length = max_question_length
        
        # Question encoder
        self.question_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Video feature encoder
        self.video_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Answer prediction
        self.answer_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers)
        )
        
        # Reasoning path predictor
        self.reasoning_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, 
                question_features: torch.Tensor,
                video_features: torch.Tensor,
                reasoning_paths: List[str],
                question_text: str = "") -> VideoQAOutput:
        """
        Answer questions about video content.
        
        Args:
            question_features: Question representation
            video_features: Video representation from GTR-Former
            reasoning_paths: Available reasoning paths
            question_text: Original question text
            
        Returns:
            VideoQAOutput with answer and reasoning
        """
        # Encode question and video
        question_encoded = self.question_encoder(question_features)
        video_encoded = self.video_encoder(video_features)
        
        # Cross-modal attention
        if question_encoded.dim() == 2:
            question_encoded = question_encoded.unsqueeze(0)
        if video_encoded.dim() == 2:
            video_encoded = video_encoded.unsqueeze(0)
        
        attended_features, attention_weights = self.cross_attention(
            question_encoded, video_encoded, video_encoded
        )
        
        # Combine features
        combined_features = torch.cat([question_encoded, attended_features], dim=-1)
        
        # Predict answer
        answer_logits = self.answer_predictor(combined_features)
        answer_probs = F.softmax(answer_logits, dim=-1)
        
        # Get most likely answer
        answer_idx = answer_probs.argmax().item()
        confidence = answer_probs.max().item()
        
        # Generate answer text (simplified)
        answer_text = f"Answer_{answer_idx}"
        
        # Predict reasoning path
        reasoning_score = torch.sigmoid(self.reasoning_predictor(attended_features))
        
        # Select reasoning path
        if reasoning_paths:
            selected_path = reasoning_paths[reasoning_score.argmax().item() % len(reasoning_paths)]
        else:
            selected_path = "No reasoning path available"
        
        # Generate supporting evidence
        supporting_evidence = [
            {
                'type': 'visual',
                'confidence': confidence,
                'description': f'Visual evidence from video features'
            },
            {
                'type': 'reasoning',
                'confidence': reasoning_score.max().item(),
                'description': f'Reasoning based on {selected_path}'
            }
        ]
        
        return VideoQAOutput(
            answer=answer_text,
            confidence=confidence,
            reasoning_path=[selected_path],
            supporting_evidence=supporting_evidence
        )


class RelationshipRecognitionHead(nn.Module):
    """Relationship recognition head for modeling long-term interactions."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_relationship_types: int = 20):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_relationship_types = num_relationship_types
        
        # Relationship types
        self.relationship_types = [
            'friend', 'family', 'colleague', 'romantic', 'stranger',
            'teacher_student', 'boss_employee', 'parent_child', 'sibling',
            'acquaintance', 'neighbor', 'classmate', 'teammate', 'partner',
            'rival', 'mentor', 'supervisor', 'peer', 'companion', 'associate'
        ]
        
        # Actor pair encoder
        self.actor_pair_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Interaction pattern encoder
        self.interaction_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal relationship modeling
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Relationship classifier
        self.relationship_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_relationship_types)
        )
        
        # Relationship feature extractor
        self.relationship_feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, 
                actor1_features: torch.Tensor,
                actor2_features: torch.Tensor,
                interaction_history: List[Dict],
                temporal_features: Optional[torch.Tensor] = None) -> RelationshipOutput:
        """
        Recognize relationship between two actors.
        
        Args:
            actor1_features: Features of first actor
            actor2_features: Features of second actor
            interaction_history: History of interactions between actors
            temporal_features: Optional temporal features
            
        Returns:
            RelationshipOutput with relationship type and confidence
        """
        # Encode actor pair
        actor_pair_features = torch.cat([actor1_features, actor2_features], dim=-1)
        pair_encoded = self.actor_pair_encoder(actor_pair_features)
        
        # Encode interaction patterns
        if interaction_history:
            interaction_features = []
            for interaction in interaction_history:
                # Extract interaction features (simplified)
                interaction_feat = torch.randn(self.feature_dim)
                interaction_encoded = self.interaction_encoder(interaction_feat)
                interaction_features.append(interaction_encoded)
            
            interaction_features = torch.stack(interaction_features)
            
            # Apply temporal encoding
            if temporal_features is not None:
                temporal_output, _ = self.temporal_encoder(temporal_features)
                interaction_features = interaction_features + temporal_output
        else:
            interaction_features = torch.zeros(1, self.hidden_dim)
        
        # Combine features
        combined_features = torch.cat([pair_encoded, interaction_features.mean(dim=0)], dim=-1)
        
        # Classify relationship
        relationship_logits = self.relationship_classifier(combined_features)
        relationship_probs = F.softmax(relationship_logits, dim=-1)
        
        # Get most likely relationship
        relationship_idx = relationship_probs.argmax().item()
        confidence = relationship_probs.max().item()
        relationship_type = self.relationship_types[relationship_idx]
        
        # Extract relationship features
        relationship_features = self.relationship_feature_extractor(combined_features)
        
        # Generate interaction pattern
        interaction_pattern = []
        for interaction in interaction_history:
            pattern = {
                'type': interaction.get('type', 'unknown'),
                'frame_id': interaction.get('frame_id', 0),
                'confidence': interaction.get('confidence', 0.5),
                'description': interaction.get('description', 'Interaction detected')
            }
            interaction_pattern.append(pattern)
        
        return RelationshipOutput(
            relationship_type=relationship_type,
            confidence=confidence,
            actor1_id=0,  # Will be set by caller
            actor2_id=1,  # Will be set by caller
            relationship_features=relationship_features,
            interaction_pattern=interaction_pattern
        )


class EventPredictionHead(nn.Module):
    """Event prediction head for anticipating future actions."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_event_types: int = 100,
                 max_prediction_horizon: int = 30):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_event_types = num_event_types
        self.max_prediction_horizon = max_prediction_horizon
        
        # Event types (simplified)
        self.event_types = [
            'walk', 'run', 'sit', 'stand', 'talk', 'listen', 'gesture',
            'point', 'wave', 'nod', 'shake_head', 'smile', 'frown',
            'laugh', 'cry', 'shout', 'whisper', 'hug', 'handshake',
            'kiss', 'fight', 'dance', 'jump', 'climb', 'fall'
        ]
        
        # Current state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Historical pattern encoder
        self.history_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Causal factor encoder
        self.causal_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Event predictor
        self.event_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_event_types)
        )
        
        # Time horizon predictor
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_prediction_horizon)
        )
        
        # Causal factor predictor
        self.causal_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # 10 causal factors
        )
        
    def forward(self, 
                current_state: torch.Tensor,
                historical_patterns: torch.Tensor,
                causal_factors: torch.Tensor,
                actor_ids: List[int]) -> EventPredictionOutput:
        """
        Predict future events based on current state and history.
        
        Args:
            current_state: Current state features
            historical_patterns: Historical behavior patterns
            causal_factors: Causal factors affecting prediction
            actor_ids: IDs of actors to predict for
            
        Returns:
            EventPredictionOutput with predicted event and details
        """
        # Encode current state
        state_encoded = self.state_encoder(current_state)
        
        # Encode historical patterns
        if historical_patterns.dim() == 2:
            historical_patterns = historical_patterns.unsqueeze(0)
        
        history_output, _ = self.history_encoder(historical_patterns)
        history_encoded = history_output.mean(dim=1)  # Average over time
        
        # Encode causal factors
        causal_encoded = self.causal_encoder(causal_factors)
        
        # Combine all features
        combined_features = torch.cat([state_encoded, history_encoded, causal_encoded], dim=-1)
        
        # Predict event type
        event_logits = self.event_predictor(combined_features)
        event_probs = F.softmax(event_logits, dim=-1)
        
        event_idx = event_probs.argmax().item()
        event_confidence = event_probs.max().item()
        predicted_event = self.event_types[event_idx % len(self.event_types)]
        
        # Predict time horizon
        time_logits = self.time_predictor(combined_features)
        time_probs = F.softmax(time_logits, dim=-1)
        
        time_horizon = time_probs.argmax().item() + 1  # 1 to max_prediction_horizon
        
        # Predict causal factors
        causal_logits = self.causal_predictor(combined_features)
        causal_probs = torch.sigmoid(causal_logits)
        
        # Select top causal factors
        top_causal_indices = causal_probs.topk(3, dim=-1)[1].squeeze()
        causal_factor_names = [
            'spatial_proximity', 'temporal_sequence', 'social_interaction',
            'emotional_state', 'physical_action', 'environmental_context',
            'previous_behavior', 'relationship_dynamics', 'external_stimulus',
            'internal_motivation'
        ]
        
        selected_causal_factors = [causal_factor_names[idx] for idx in top_causal_indices]
        
        return EventPredictionOutput(
            predicted_event=predicted_event,
            confidence=event_confidence,
            time_horizon=time_horizon,
            affected_actors=actor_ids,
            causal_factors=selected_causal_factors
        )


class TaskSpecificHeads(nn.Module):
    """Combined task-specific output heads."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Initialize task heads
        self.videoqa_head = VideoQAHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        self.relationship_head = RelationshipRecognitionHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        self.event_prediction_head = EventPredictionHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        # Shared feature projection
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, 
                gtr_output: Dict,
                task_type: str,
                **kwargs) -> Dict:
        """
        Apply task-specific heads based on task type.
        
        Args:
            gtr_output: Output from GTR-Former
            task_type: Type of task ('videoqa', 'relationship', 'event_prediction')
            **kwargs: Task-specific arguments
            
        Returns:
            Task-specific output
        """
        # Extract features from GTR-Former output
        node_features = gtr_output['node_features']
        reasoning_paths = gtr_output.get('reasoning_paths', [])
        
        # Project features
        projected_features = self.feature_projection(node_features)
        
        if task_type == 'videoqa':
            return self._videoqa_forward(projected_features, reasoning_paths, **kwargs)
        elif task_type == 'relationship':
            return self._relationship_forward(projected_features, **kwargs)
        elif task_type == 'event_prediction':
            return self._event_prediction_forward(projected_features, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _videoqa_forward(self, 
                        features: torch.Tensor,
                        reasoning_paths: List,
                        question_features: torch.Tensor,
                        question_text: str = "") -> Dict:
        """Forward pass for VideoQA task."""
        # Use average of node features as video representation
        video_features = features.mean(dim=0)
        
        # Get reasoning path strings
        reasoning_path_strings = [f"Path_{i}" for i in range(len(reasoning_paths))]
        
        # Apply VideoQA head
        qa_output = self.videoqa_head(
            question_features=question_features,
            video_features=video_features,
            reasoning_paths=reasoning_path_strings,
            question_text=question_text
        )
        
        return {
            'task_type': 'videoqa',
            'output': qa_output
        }
    
    def _relationship_forward(self, 
                            features: torch.Tensor,
                            actor1_id: int = 0,
                            actor2_id: int = 1,
                            interaction_history: List[Dict] = None) -> Dict:
        """Forward pass for relationship recognition task."""
        if interaction_history is None:
            interaction_history = []
        
        # Get actor features
        if actor1_id < features.shape[0] and actor2_id < features.shape[0]:
            actor1_features = features[actor1_id]
            actor2_features = features[actor2_id]
        else:
            # Use first two features if actor IDs are out of range
            actor1_features = features[0]
            actor2_features = features[1] if features.shape[0] > 1 else features[0]
        
        # Apply relationship head
        relationship_output = self.relationship_head(
            actor1_features=actor1_features,
            actor2_features=actor2_features,
            interaction_history=interaction_history
        )
        
        # Update actor IDs
        relationship_output.actor1_id = actor1_id
        relationship_output.actor2_id = actor2_id
        
        return {
            'task_type': 'relationship',
            'output': relationship_output
        }
    
    def _event_prediction_forward(self, 
                                features: torch.Tensor,
                                actor_ids: List[int] = None,
                                historical_patterns: torch.Tensor = None,
                                causal_factors: torch.Tensor = None) -> Dict:
        """Forward pass for event prediction task."""
        if actor_ids is None:
            actor_ids = [0]
        
        if historical_patterns is None:
            historical_patterns = features.unsqueeze(0)  # Use current features as history
        
        if causal_factors is None:
            causal_factors = features.mean(dim=0)  # Use average features as causal factors
        
        # Use average of features as current state
        current_state = features.mean(dim=0)
        
        # Apply event prediction head
        event_output = self.event_prediction_head(
            current_state=current_state,
            historical_patterns=historical_patterns,
            causal_factors=causal_factors,
            actor_ids=actor_ids
        )
        
        return {
            'task_type': 'event_prediction',
            'output': event_output
        }
    
    def get_task_specific_features(self, 
                                 gtr_output: Dict,
                                 task_type: str) -> torch.Tensor:
        """Extract task-specific features from GTR-Former output."""
        node_features = gtr_output['node_features']
        
        if task_type == 'videoqa':
            # For VideoQA, use global video representation
            return node_features.mean(dim=0)
        elif task_type == 'relationship':
            # For relationship, use pairwise features
            if node_features.shape[0] >= 2:
                return torch.cat([node_features[0], node_features[1]], dim=-1)
            else:
                return node_features[0]
        elif task_type == 'event_prediction':
            # For event prediction, use temporal features
            return node_features.mean(dim=0)
        else:
            return node_features.mean(dim=0)
