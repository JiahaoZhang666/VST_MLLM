"""
SpatioTemporal Actor Graph (STAG) Module for VideoSpatioTemporal Framework

This module implements the SpatioTemporal Actor Graph with three types of edges:
- Temporal edges: Continuity of the same person across different time points
- Spatial edges: Relationships between people in the same frame
- Speech edges: Who is talking to whom

Features:
- Dynamic graph construction with sliding window
- Heterogeneous edge types with learnable features
- Multi-modal edge features (position, appearance, speech, gaze)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from .entity_extraction import EntityDetection, EntityTrack
from .constants import MAX_ENTITIES, MAX_TEMPORAL_EDGES, SPATIAL_DISTANCE_THRESHOLD


class EdgeType(Enum):
    """Types of edges in the SpatioTemporal Actor Graph."""
    TEMPORAL = "temporal"      # Same person across time
    SPATIAL = "spatial"        # People in same frame
    SPEECH = "speech"          # Who is talking to whom


@dataclass
class ActorNode:
    """Represents an actor (person) in the graph."""
    node_id: str  # Format: "person_{track_id}_{frame_id}"
    track_id: int
    frame_id: int
    entity_detection: EntityDetection
    features: torch.Tensor
    position: torch.Tensor
    is_speaking: bool
    emotion: Optional[str] = None


@dataclass
class ActorEdge:
    """Represents an edge between actors in the graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    features: torch.Tensor
    confidence: float
    metadata: Dict  # Additional edge-specific information


class HeterogeneousGraphAttention(nn.Module):
    """Heterogeneous graph attention for different edge types."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 num_heads: int = 8,
                 num_edge_types: int = 3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.head_dim = feature_dim // num_heads
        
        # Edge-type specific attention weights
        self.edge_type_embeddings = nn.Embedding(num_edge_types, feature_dim)
        
        # Multi-head attention
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, 
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_types: torch.Tensor,
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply heterogeneous graph attention.
        
        Args:
            node_features: [num_nodes, feature_dim]
            edge_features: [num_edges, feature_dim]
            edge_types: [num_edges] edge type indices
            adjacency_matrix: [num_nodes, num_nodes] sparse adjacency matrix
            
        Returns:
            Updated node features
        """
        batch_size, num_nodes, feature_dim = node_features.shape
        
        # Get edge type embeddings
        edge_type_emb = self.edge_type_embeddings(edge_types)  # [num_edges, feature_dim]
        
        # Add edge type information to edge features
        enhanced_edge_features = edge_features + edge_type_emb
        
        # Multi-head attention
        Q = self.query(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.key(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.value(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply adjacency mask
        attention_scores = attention_scores.masked_fill(adjacency_matrix.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended_values = attended_values.view(batch_size, num_nodes, feature_dim)
        output = self.output_projection(attended_values)
        
        # Residual connection and layer norm
        output = self.layer_norm(node_features + output)
        
        return output


class SpatioTemporalActorGraph(nn.Module):
    """SpatioTemporal Actor Graph with dynamic construction."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 max_entities: int = MAX_ENTITIES,
                 max_temporal_edges: int = MAX_TEMPORAL_EDGES,
                 spatial_threshold: float = SPATIAL_DISTANCE_THRESHOLD,
                 temporal_window: int = 16,
                 num_attention_heads: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_entities = max_entities
        self.max_temporal_edges = max_temporal_edges
        self.spatial_threshold = spatial_threshold
        self.temporal_window = temporal_window
        self.num_attention_heads = num_attention_heads
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Embedding(3, feature_dim)  # temporal, spatial, speech
        
        # Edge feature extractors for different edge types
        self.temporal_edge_extractor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, feature_dim),  # +2 for temporal features
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.spatial_edge_extractor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, feature_dim),  # +4 for spatial features
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.speech_edge_extractor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, feature_dim),  # +3 for speech features
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Heterogeneous graph attention
        self.graph_attention = HeterogeneousGraphAttention(
            feature_dim=feature_dim,
            num_heads=num_attention_heads,
            num_edge_types=3
        )
        
        # Node feature update
        self.node_update = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Graph state tracking
        self.nodes = {}  # node_id -> ActorNode
        self.edges = []  # List of ActorEdge
        self.temporal_memory = {}  # track_id -> recent nodes
        
    def forward(self, 
                entity_detections: List[List[EntityDetection]],
                audio_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Build and process the SpatioTemporal Actor Graph.
        
        Args:
            entity_detections: List of entity detections per frame
            audio_features: Optional audio features for speech analysis
            
        Returns:
            Dictionary containing graph information and processed features
        """
        # Build graph nodes
        self._build_nodes(entity_detections)
        
        # Construct edges
        temporal_edges = self._construct_temporal_edges()
        spatial_edges = self._construct_spatial_edges()
        speech_edges = self._construct_speech_edges(audio_features)
        
        # Combine all edges
        all_edges = temporal_edges + spatial_edges + speech_edges
        self.edges = all_edges
        
        # Build graph structure
        graph = self._build_networkx_graph()
        
        # Extract graph features
        node_features, edge_features, adjacency_matrix, edge_types = self._extract_graph_features()
        
        # Apply graph attention
        if len(node_features) > 0:
            enhanced_node_features = self.graph_attention(
                node_features.unsqueeze(0),
                edge_features,
                edge_types,
                adjacency_matrix.unsqueeze(0)
            ).squeeze(0)
        else:
            enhanced_node_features = node_features
        
        return {
            'graph': graph,
            'nodes': self.nodes,
            'edges': all_edges,
            'node_features': enhanced_node_features,
            'edge_features': edge_features,
            'adjacency_matrix': adjacency_matrix,
            'edge_types': edge_types
        }
    
    def _build_nodes(self, entity_detections: List[List[EntityDetection]]):
        """Build graph nodes from entity detections."""
        self.nodes = {}
        
        for frame_idx, frame_detections in enumerate(entity_detections):
            for detection in frame_detections:
                # Create node ID
                node_id = f"person_{detection.entity_id}_{frame_idx}"
                
                # Create actor node
                actor_node = ActorNode(
                    node_id=node_id,
                    track_id=detection.entity_id,
                    frame_id=frame_idx,
                    entity_detection=detection,
                    features=detection.features,
                    position=detection.position,
                    is_speaking=detection.is_speaking,
                    emotion=detection.emotion
                )
                
                self.nodes[node_id] = actor_node
                
                # Update temporal memory
                if detection.entity_id not in self.temporal_memory:
                    self.temporal_memory[detection.entity_id] = []
                self.temporal_memory[detection.entity_id].append(node_id)
                
                # Keep only recent nodes in memory
                if len(self.temporal_memory[detection.entity_id]) > self.temporal_window:
                    self.temporal_memory[detection.entity_id] = self.temporal_memory[detection.entity_id][-self.temporal_window:]
    
    def _construct_temporal_edges(self) -> List[ActorEdge]:
        """Construct temporal edges for same person across time."""
        temporal_edges = []
        
        for track_id, node_ids in self.temporal_memory.items():
            if len(node_ids) < 2:
                continue
            
            # Create edges between consecutive time points
            for i in range(len(node_ids) - 1):
                source_id = node_ids[i]
                target_id = node_ids[i + 1]
                
                source_node = self.nodes[source_id]
                target_node = self.nodes[target_id]
                
                # Calculate temporal features
                temporal_gap = target_node.frame_id - source_node.frame_id
                temporal_position = target_node.frame_id / max(n.frame_id for n in self.nodes.values())
                
                # Extract edge features
                temporal_features = torch.cat([
                    source_node.features,
                    target_node.features,
                    torch.tensor([temporal_gap, temporal_position], device=source_node.features.device)
                ])
                
                edge_features = self.temporal_edge_extractor(temporal_features)
                confidence = torch.sigmoid(edge_features.mean()).item()
                
                edge = ActorEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=EdgeType.TEMPORAL,
                    features=edge_features,
                    confidence=confidence,
                    metadata={
                        'temporal_gap': temporal_gap,
                        'temporal_position': temporal_position
                    }
                )
                
                temporal_edges.append(edge)
        
        return temporal_edges
    
    def _construct_spatial_edges(self) -> List[ActorEdge]:
        """Construct spatial edges between people in the same frame."""
        spatial_edges = []
        
        # Group nodes by frame
        frame_nodes = {}
        for node_id, node in self.nodes.items():
            if node.frame_id not in frame_nodes:
                frame_nodes[node.frame_id] = []
            frame_nodes[node.frame_id].append(node)
        
        # Create spatial edges within each frame
        for frame_id, frame_node_list in frame_nodes.items():
            for i, source_node in enumerate(frame_node_list):
                for j, target_node in enumerate(frame_node_list):
                    if i == j:
                        continue
                    
                    # Calculate spatial relationship
                    distance = torch.norm(source_node.position - target_node.position)
                    
                    # Only create edges for nearby people
                    if distance > self.spatial_threshold:
                        continue
                    
                    # Calculate spatial features
                    dx = target_node.position[0] - source_node.position[0]
                    dy = target_node.position[1] - source_node.position[1]
                    angle = torch.atan2(dy, dx)
                    
                    # Extract edge features
                    spatial_features = torch.cat([
                        source_node.features,
                        target_node.features,
                        torch.tensor([dx, dy, distance, angle], device=source_node.features.device)
                    ])
                    
                    edge_features = self.spatial_edge_extractor(spatial_features)
                    confidence = torch.sigmoid(edge_features.mean()).item()
                    
                    edge = ActorEdge(
                        source_id=source_node.node_id,
                        target_id=target_node.node_id,
                        edge_type=EdgeType.SPATIAL,
                        features=edge_features,
                        confidence=confidence,
                        metadata={
                            'distance': distance.item(),
                            'angle': angle.item(),
                            'dx': dx.item(),
                            'dy': dy.item()
                        }
                    )
                    
                    spatial_edges.append(edge)
        
        return spatial_edges
    
    def _construct_speech_edges(self, audio_features: Optional[torch.Tensor] = None) -> List[ActorEdge]:
        """Construct speech edges based on who is talking to whom."""
        speech_edges = []
        
        if audio_features is None:
            return speech_edges
        
        # Find speaking nodes
        speaking_nodes = [node for node in self.nodes.values() if node.is_speaking]
        non_speaking_nodes = [node for node in self.nodes.values() if not node.is_speaking]
        
        # Create speech edges from speakers to listeners
        for speaker in speaking_nodes:
            for listener in non_speaking_nodes:
                # Only create edges within the same frame or nearby frames
                frame_gap = abs(speaker.frame_id - listener.frame_id)
                if frame_gap > 3:  # Max 3 frame gap for speech edges
                    continue
                
                # Calculate speech features
                distance = torch.norm(speaker.position - listener.position)
                gaze_direction = self._calculate_gaze_direction(speaker, listener)
                
                # Extract edge features
                speech_features = torch.cat([
                    speaker.features,
                    listener.features,
                    torch.tensor([distance, gaze_direction, frame_gap], device=speaker.features.device)
                ])
                
                edge_features = self.speech_edge_extractor(speech_features)
                confidence = torch.sigmoid(edge_features.mean()).item()
                
                # Only create edges with high confidence
                if confidence > 0.5:
                    edge = ActorEdge(
                        source_id=speaker.node_id,
                        target_id=listener.node_id,
                        edge_type=EdgeType.SPEECH,
                        features=edge_features,
                        confidence=confidence,
                        metadata={
                            'distance': distance.item(),
                            'gaze_direction': gaze_direction,
                            'frame_gap': frame_gap,
                            'speaker_emotion': speaker.emotion
                        }
                    )
                    
                    speech_edges.append(edge)
        
        return speech_edges
    
    def _calculate_gaze_direction(self, speaker: ActorNode, listener: ActorNode) -> float:
        """Calculate gaze direction from speaker to listener."""
        # Simplified gaze direction calculation
        # In practice, this would use gaze estimation models
        
        # Calculate angle from speaker to listener
        dx = listener.position[0] - speaker.position[0]
        dy = listener.position[1] - speaker.position[1]
        angle = torch.atan2(dy, dx)
        
        # Normalize to [0, 1]
        normalized_angle = (angle + np.pi) / (2 * np.pi)
        
        return normalized_angle.item()
    
    def _build_networkx_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph representation."""
        graph = nx.MultiDiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            graph.add_node(
                node_id,
                track_id=node.track_id,
                frame_id=node.frame_id,
                position=node.position.cpu().numpy(),
                features=node.features.cpu().numpy(),
                is_speaking=node.is_speaking,
                emotion=node.emotion
            )
        
        # Add edges
        for edge in self.edges:
            graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                features=edge.features.cpu().numpy(),
                confidence=edge.confidence,
                metadata=edge.metadata
            )
        
        return graph
    
    def _extract_graph_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract feature matrices from the graph."""
        if not self.nodes:
            return torch.empty(0, self.feature_dim), torch.empty(0, self.feature_dim), torch.zeros(0, 0), torch.empty(0)
        
        # Node features
        node_ids = list(self.nodes.keys())
        node_features = torch.stack([self.nodes[node_id].features for node_id in node_ids])
        
        # Edge features
        if self.edges:
            edge_features = torch.stack([edge.features for edge in self.edges])
            edge_types = torch.tensor([edge.edge_type.value for edge in self.edges])
        else:
            edge_features = torch.empty(0, self.feature_dim)
            edge_types = torch.empty(0)
        
        # Adjacency matrix
        num_nodes = len(node_ids)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes)
        
        # Create node ID to index mapping
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        for edge in self.edges:
            if edge.source_id in node_id_to_idx and edge.target_id in node_id_to_idx:
                source_idx = node_id_to_idx[edge.source_id]
                target_idx = node_id_to_idx[edge.target_id]
                adjacency_matrix[source_idx, target_idx] = edge.confidence
        
        return node_features, edge_features, adjacency_matrix, edge_types
    
    def get_actor_trajectories(self) -> Dict[int, List[ActorNode]]:
        """Get trajectories for each actor."""
        trajectories = {}
        
        for node in self.nodes.values():
            track_id = node.track_id
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append(node)
        
        # Sort by frame_id
        for track_id in trajectories:
            trajectories[track_id] = sorted(trajectories[track_id], key=lambda x: x.frame_id)
        
        return trajectories
    
    def get_speaking_patterns(self) -> List[Dict]:
        """Get speaking patterns and interactions."""
        speaking_patterns = []
        
        speech_edges = [edge for edge in self.edges if edge.edge_type == EdgeType.SPEECH]
        
        for edge in speech_edges:
            speaker_node = self.nodes[edge.source_id]
            listener_node = self.nodes[edge.target_id]
            
            pattern = {
                'speaker_id': speaker_node.track_id,
                'listener_id': listener_node.track_id,
                'frame_id': speaker_node.frame_id,
                'confidence': edge.confidence,
                'speaker_emotion': speaker_node.emotion,
                'distance': edge.metadata['distance'],
                'gaze_direction': edge.metadata['gaze_direction']
            }
            
            speaking_patterns.append(pattern)
        
        return speaking_patterns
    
    def get_spatial_relationships(self) -> List[Dict]:
        """Get spatial relationships between actors."""
        spatial_relationships = []
        
        spatial_edges = [edge for edge in self.edges if edge.edge_type == EdgeType.SPATIAL]
        
        for edge in spatial_edges:
            source_node = self.nodes[edge.source_id]
            target_node = self.nodes[edge.target_id]
            
            relationship = {
                'actor1_id': source_node.track_id,
                'actor2_id': target_node.track_id,
                'frame_id': source_node.frame_id,
                'distance': edge.metadata['distance'],
                'angle': edge.metadata['angle'],
                'confidence': edge.confidence
            }
            
            spatial_relationships.append(relationship)
        
        return spatial_relationships
