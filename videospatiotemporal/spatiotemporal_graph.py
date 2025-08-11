"""
Spatiotemporal Graph Builder Module for VideoSpatioTemporal Framework

This module constructs dynamic spatiotemporal graphs that capture the evolving relationships
between entities in videos, enabling comprehensive understanding of spatial configurations
and temporal evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from .constants import (
    MAX_ENTITIES, MAX_TEMPORAL_EDGES, SPATIAL_DISTANCE_THRESHOLD,
    TEMPORAL_WINDOW_SIZE, GRAPH_ATTENTION_HEADS, GRAPH_HIDDEN_DIM,
    TEMPORAL_SCALES, SPATIAL_SCALES
)


@dataclass
class EntityNode:
    """Represents an entity in the spatiotemporal graph."""
    entity_id: int
    frame_id: int
    position: torch.Tensor  # (x, y) coordinates
    features: torch.Tensor  # Entity features
    confidence: float
    entity_type: str = "person"  # person, object, etc.


@dataclass
class SpatiotemporalEdge:
    """Represents a relationship between entities across space and time."""
    source_id: int
    target_id: int
    source_frame: int
    target_frame: int
    relationship_type: str
    confidence: float
    spatial_distance: float
    temporal_distance: int
    features: torch.Tensor


class SpatiotemporalGraphBuilder(nn.Module):
    """
    Builds and maintains dynamic spatiotemporal graphs for video understanding.
    
    Features:
    - Multi-scale spatial relationship modeling
    - Temporal hyperedge construction
    - Dynamic graph evolution
    - Learnable edge construction
    """
    
    def __init__(self, 
                 feature_dim: int = 512,
                 max_entities: int = MAX_ENTITIES,
                 max_temporal_edges: int = MAX_TEMPORAL_EDGES,
                 spatial_threshold: float = SPATIAL_DISTANCE_THRESHOLD,
                 temporal_window: int = TEMPORAL_WINDOW_SIZE,
                 num_heads: int = GRAPH_ATTENTION_HEADS,
                 hidden_dim: int = GRAPH_HIDDEN_DIM):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_entities = max_entities
        self.max_temporal_edges = max_temporal_edges
        self.spatial_threshold = spatial_threshold
        self.temporal_window = temporal_window
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Learnable edge construction
        self.edge_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, hidden_dim),  # +4 for spatial features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # (dx, dy, distance, angle)
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Multi-scale graph transformer
        self.graph_transformer = MultiScaleGraphTransformer(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=3
        )
        
        # Temporal hyperedge construction
        self.hyperedge_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, hidden_dim),  # +2 for temporal features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Graph state tracking
        self.entity_tracker = EntityTracker(max_entities=max_entities)
        self.relationship_memory = RelationshipMemory(max_edges=max_temporal_edges)
        
    def forward(self, 
                video_features: torch.Tensor,
                entity_detections: List[List[EntityNode]],
                masks: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Build spatiotemporal graph from video features and entity detections.
        
        Args:
            video_features: Video frame features (T, C, H, W)
            entity_detections: List of entity detections per frame
            masks: Optional object masks
            audio_features: Optional audio features for alignment
            
        Returns:
            Dictionary containing:
            - graph: NetworkX graph representation
            - node_features: Node feature matrix
            - edge_features: Edge feature matrix
            - adjacency_matrix: Adjacency matrix
            - temporal_hyperedges: Temporal relationship hyperedges
        """
        batch_size = video_features.shape[0] if video_features.dim() > 3 else 1
        
        # Initialize graph
        graph = nx.MultiDiGraph()
        all_nodes = []
        all_edges = []
        
        # Process each frame
        for frame_idx, frame_entities in enumerate(entity_detections):
            # Add nodes for current frame
            frame_nodes = self._process_frame_entities(
                frame_entities, frame_idx, video_features[frame_idx] if video_features.dim() > 3 else video_features
            )
            all_nodes.extend(frame_nodes)
            
            # Add spatial edges within frame
            spatial_edges = self._construct_spatial_edges(frame_nodes, frame_idx)
            all_edges.extend(spatial_edges)
            
            # Add temporal edges across frames
            if frame_idx > 0:
                temporal_edges = self._construct_temporal_edges(
                    all_nodes, frame_idx, self.temporal_window
                )
                all_edges.extend(temporal_edges)
        
        # Construct temporal hyperedges
        temporal_hyperedges = self._construct_temporal_hyperedges(all_nodes, all_edges)
        
        # Build graph structure
        graph = self._build_networkx_graph(all_nodes, all_edges, temporal_hyperedges)
        
        # Extract graph features
        node_features, edge_features, adjacency_matrix = self._extract_graph_features(
            graph, all_nodes, all_edges
        )
        
        # Apply graph transformer
        enhanced_node_features = self.graph_transformer(
            node_features, adjacency_matrix, edge_features
        )
        
        return {
            'graph': graph,
            'node_features': enhanced_node_features,
            'edge_features': edge_features,
            'adjacency_matrix': adjacency_matrix,
            'temporal_hyperedges': temporal_hyperedges,
            'entities': all_nodes,
            'relationships': all_edges
        }
    
    def _process_frame_entities(self, 
                               frame_entities: List[EntityNode],
                               frame_idx: int,
                               frame_features: torch.Tensor) -> List[EntityNode]:
        """Process entities detected in a single frame."""
        processed_nodes = []
        
        for entity in frame_entities:
            # Extract spatial features from frame
            spatial_features = self._extract_spatial_features(entity.position, frame_features)
            
            # Update entity features
            enhanced_features = torch.cat([entity.features, spatial_features], dim=-1)
            
            # Create enhanced node
            enhanced_node = EntityNode(
                entity_id=entity.entity_id,
                frame_id=frame_idx,
                position=entity.position,
                features=enhanced_features,
                confidence=entity.confidence,
                entity_type=entity.entity_type
            )
            
            processed_nodes.append(enhanced_node)
        
        return processed_nodes
    
    def _extract_spatial_features(self, 
                                 position: torch.Tensor,
                                 frame_features: torch.Tensor) -> torch.Tensor:
        """Extract spatial context features from frame."""
        # Simple spatial feature extraction (can be enhanced)
        h, w = frame_features.shape[-2:]
        x, y = position[0], position[1]
        
        # Normalize coordinates
        x_norm = x / w
        y_norm = y / h
        
        # Extract local patch features
        patch_size = 16
        x_start = max(0, int(x - patch_size // 2))
        y_start = max(0, int(y - patch_size // 2))
        x_end = min(w, x_start + patch_size)
        y_end = min(h, y_start + patch_size)
        
        if x_end > x_start and y_end > y_start:
            local_features = frame_features[..., y_start:y_end, x_start:x_end]
            local_features = F.adaptive_avg_pool2d(local_features, (1, 1)).flatten()
        else:
            local_features = torch.zeros(self.feature_dim // 4, device=position.device)
        
        # Combine position and local features
        spatial_features = torch.cat([
            torch.tensor([x_norm, y_norm], device=position.device),
            local_features
        ])
        
        return spatial_features
    
    def _construct_spatial_edges(self, 
                                frame_nodes: List[EntityNode],
                                frame_idx: int) -> List[SpatiotemporalEdge]:
        """Construct spatial relationships within a frame."""
        edges = []
        
        for i, node_i in enumerate(frame_nodes):
            for j, node_j in enumerate(frame_nodes):
                if i == j:
                    continue
                
                # Calculate spatial relationship
                pos_i = node_i.position
                pos_j = node_j.position
                
                distance = torch.norm(pos_i - pos_j)
                dx = pos_j[0] - pos_i[0]
                dy = pos_j[1] - pos_i[1]
                angle = torch.atan2(dy, dx)
                
                # Predict relationship confidence
                relationship_features = torch.cat([
                    node_i.features,
                    node_j.features,
                    torch.tensor([dx, dy, distance, angle], device=pos_i.device)
                ])
                
                confidence = self.edge_predictor(relationship_features)
                
                # Only add edges above threshold
                if confidence > self.spatial_threshold:
                    edge = SpatiotemporalEdge(
                        source_id=node_i.entity_id,
                        target_id=node_j.entity_id,
                        source_frame=frame_idx,
                        target_frame=frame_idx,
                        relationship_type="spatial_proximity",
                        confidence=confidence.item(),
                        spatial_distance=distance.item(),
                        temporal_distance=0,
                        features=relationship_features
                    )
                    edges.append(edge)
        
        return edges
    
    def _construct_temporal_edges(self, 
                                 all_nodes: List[EntityNode],
                                 current_frame: int,
                                 window_size: int) -> List[SpatiotemporalEdge]:
        """Construct temporal relationships across frames."""
        edges = []
        
        # Find nodes in temporal window
        window_start = max(0, current_frame - window_size)
        current_nodes = [n for n in all_nodes if n.frame_id == current_frame]
        window_nodes = [n for n in all_nodes if window_start <= n.frame_id < current_frame]
        
        for current_node in current_nodes:
            for window_node in window_nodes:
                # Check if same entity (simple heuristic)
                if current_node.entity_id == window_node.entity_id:
                    temporal_distance = current_frame - window_node.frame_id
                    
                    # Predict temporal relationship
                    temporal_features = torch.cat([
                        current_node.features,
                        window_node.features,
                        torch.tensor([temporal_distance, 1.0], device=current_node.features.device)
                    ])
                    
                    confidence = self.hyperedge_predictor(temporal_features)
                    
                    if confidence > 0.5:  # Temporal relationship threshold
                        edge = SpatiotemporalEdge(
                            source_id=window_node.entity_id,
                            target_id=current_node.entity_id,
                            source_frame=window_node.frame_id,
                            target_frame=current_frame,
                            relationship_type="temporal_sequence",
                            confidence=confidence.item(),
                            spatial_distance=torch.norm(current_node.position - window_node.position).item(),
                            temporal_distance=temporal_distance,
                            features=temporal_features
                        )
                        edges.append(edge)
        
        return edges
    
    def _construct_temporal_hyperedges(self, 
                                     nodes: List[EntityNode],
                                     edges: List[SpatiotemporalEdge]) -> List[Dict]:
        """Construct temporal hyperedges for capturing complex temporal patterns."""
        hyperedges = []
        
        # Group edges by relationship type
        edge_groups = {}
        for edge in edges:
            if edge.relationship_type not in edge_groups:
                edge_groups[edge.relationship_type] = []
            edge_groups[edge.relationship_type].append(edge)
        
        # Create hyperedges for each group
        for rel_type, rel_edges in edge_groups.items():
            if len(rel_edges) > 1:
                # Find connected components
                connected_components = self._find_connected_components(rel_edges)
                
                for component in connected_components:
                    if len(component) > 2:  # Only create hyperedges for complex patterns
                        hyperedge = {
                            'type': rel_type,
                            'edges': component,
                            'nodes': list(set([e.source_id for e in component] + [e.target_id for e in component])),
                            'frames': list(set([e.source_frame for e in component] + [e.target_frame for e in component])),
                            'confidence': np.mean([e.confidence for e in component])
                        }
                        hyperedges.append(hyperedge)
        
        return hyperedges
    
    def _find_connected_components(self, edges: List[SpatiotemporalEdge]) -> List[List[SpatiotemporalEdge]]:
        """Find connected components in a set of edges."""
        # Simple connected component finding
        components = []
        visited = set()
        
        for edge in edges:
            if edge.source_id in visited or edge.target_id in visited:
                continue
            
            component = [edge]
            visited.add(edge.source_id)
            visited.add(edge.target_id)
            
            # Find all connected edges
            changed = True
            while changed:
                changed = False
                for other_edge in edges:
                    if other_edge in component:
                        continue
                    
                    if (other_edge.source_id in visited or 
                        other_edge.target_id in visited):
                        component.append(other_edge)
                        visited.add(other_edge.source_id)
                        visited.add(other_edge.target_id)
                        changed = True
            
            components.append(component)
        
        return components
    
    def _build_networkx_graph(self, 
                             nodes: List[EntityNode],
                             edges: List[SpatiotemporalEdge],
                             hyperedges: List[Dict]) -> nx.MultiDiGraph:
        """Build NetworkX graph from nodes and edges."""
        graph = nx.MultiDiGraph()
        
        # Add nodes
        for node in nodes:
            graph.add_node(
                f"{node.entity_id}_{node.frame_id}",
                entity_id=node.entity_id,
                frame_id=node.frame_id,
                position=node.position.cpu().numpy(),
                features=node.features.cpu().numpy(),
                confidence=node.confidence,
                entity_type=node.entity_type
            )
        
        # Add edges
        for edge in edges:
            source_node = f"{edge.source_id}_{edge.source_frame}"
            target_node = f"{edge.target_id}_{edge.target_frame}"
            
            graph.add_edge(
                source_node,
                target_node,
                relationship_type=edge.relationship_type,
                confidence=edge.confidence,
                spatial_distance=edge.spatial_distance,
                temporal_distance=edge.temporal_distance,
                features=edge.features.cpu().numpy()
            )
        
        # Add hyperedge information as graph attributes
        graph.graph['hyperedges'] = hyperedges
        
        return graph
    
    def _extract_graph_features(self, 
                               graph: nx.MultiDiGraph,
                               nodes: List[EntityNode],
                               edges: List[SpatiotemporalEdge]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract feature matrices from the graph."""
        # Node features
        node_features = torch.stack([node.features for node in nodes])
        
        # Edge features
        if edges:
            edge_features = torch.stack([edge.features for edge in edges])
        else:
            edge_features = torch.empty(0, self.feature_dim + 4, device=node_features.device)
        
        # Adjacency matrix
        num_nodes = len(nodes)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=node_features.device)
        
        for edge in edges:
            source_idx = next(i for i, n in enumerate(nodes) 
                            if n.entity_id == edge.source_id and n.frame_id == edge.source_frame)
            target_idx = next(i for i, n in enumerate(nodes) 
                            if n.entity_id == edge.target_id and n.frame_id == edge.target_frame)
            adjacency_matrix[source_idx, target_idx] = edge.confidence
        
        return node_features, edge_features, adjacency_matrix


class MultiScaleGraphTransformer(nn.Module):
    """Multi-scale graph transformer for processing spatiotemporal graphs."""
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Feature projection
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Multi-scale layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, 
                node_features: torch.Tensor,
                adjacency_matrix: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process graph through multi-scale transformer layers."""
        x = self.input_projection(node_features)
        
        for i, layer in enumerate(self.layers):
            x = x + layer(x, adjacency_matrix, edge_features)
            x = self.layer_norms[i](x)
        
        x = self.output_projection(x)
        return x


class GraphTransformerLayer(nn.Module):
    """Single graph transformer layer."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                x: torch.Tensor,
                adjacency_matrix: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention with graph structure
        attn_output, _ = self.attention(x, x, x, attn_mask=adjacency_matrix)
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        
        return x


class EntityTracker(nn.Module):
    """Tracks entities across frames for consistent identification."""
    
    def __init__(self, max_entities: int = MAX_ENTITIES):
        super().__init__()
        self.max_entities = max_entities
        self.entity_history = {}
        self.next_entity_id = 0
    
    def update(self, frame_entities: List[EntityNode], frame_id: int):
        """Update entity tracking for a new frame."""
        # Simple tracking based on spatial proximity
        # In practice, this should use more sophisticated tracking algorithms
        
        for entity in frame_entities:
            # Find closest entity in previous frame
            best_match = None
            best_distance = float('inf')
            
            for prev_id, prev_entities in self.entity_history.items():
                if prev_entities:
                    last_entity = prev_entities[-1]
                    distance = torch.norm(entity.position - last_entity.position)
                    
                    if distance < best_distance and distance < 100:  # Threshold
                        best_distance = distance
                        best_match = prev_id
            
            if best_match is not None:
                # Update existing entity
                entity.entity_id = best_match
                if best_match in self.entity_history:
                    self.entity_history[best_match].append(entity)
                else:
                    self.entity_history[best_match] = [entity]
            else:
                # New entity
                entity.entity_id = self.next_entity_id
                self.entity_history[self.next_entity_id] = [entity]
                self.next_entity_id += 1


class RelationshipMemory(nn.Module):
    """Maintains memory of relationships across time."""
    
    def __init__(self, max_edges: int = MAX_TEMPORAL_EDGES):
        super().__init__()
        self.max_edges = max_edges
        self.relationship_history = []
    
    def add_relationships(self, relationships: List[SpatiotemporalEdge]):
        """Add new relationships to memory."""
        self.relationship_history.extend(relationships)
        
        # Maintain memory size
        if len(self.relationship_history) > self.max_edges:
            # Remove oldest relationships
            self.relationship_history = self.relationship_history[-self.max_edges:]
    
    def get_recent_relationships(self, window_size: int = 10) -> List[SpatiotemporalEdge]:
        """Get relationships from recent frames."""
        if not self.relationship_history:
            return []
        
        recent_frame = max(r.source_frame for r in self.relationship_history)
        threshold = recent_frame - window_size
        
        return [r for r in self.relationship_history if r.source_frame >= threshold]
