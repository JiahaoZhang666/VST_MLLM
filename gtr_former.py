"""
Graph-Temporal Reasoning Transformer (GTR-Former) Module for VideoSpatioTemporal Framework

This module implements the advanced reasoning architecture with:
- Entity Flow: Modeling entity-specific temporal patterns
- Time Flow: Modeling temporal evolution across all entities
- Cross-Flow Attention: Interaction between different flows
- Mixture-of-Experts (MoE) for dynamic routing
- Hierarchical Path Reasoning (HPCR) for multi-hop inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .actor_graph import ActorNode, ActorEdge, EdgeType
from .constants import GRAPH_HIDDEN_DIM, GRAPH_ATTENTION_HEADS


@dataclass
class ReasoningPath:
    """Represents a reasoning path in hierarchical path reasoning."""
    path_id: int
    nodes: List[str]  # Node IDs in the path
    path_type: str    # "causal", "spatial", "temporal", "social"
    confidence: float
    features: torch.Tensor


class MixtureOfExperts(nn.Module):
    """Mixture-of-Experts (MoE) layer for dynamic routing."""
    
    def __init__(self, 
                 input_dim: int,
                 expert_dim: int,
                 num_experts: int = 8,
                 num_selected_experts: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MoE layer with dynamic routing.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Output features
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Compute gating weights
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Select top-k experts
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_probs, self.num_selected_experts, dim=-1)
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Apply experts
        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch_size, seq_len, num_experts, input_dim]
        
        # Gather selected expert outputs
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, seq_len, self.num_selected_experts)
        seq_indices = torch.arange(seq_len).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_selected_experts)
        
        selected_outputs = expert_outputs[batch_indices, seq_indices, top_k_indices]  # [batch_size, seq_len, num_selected_experts, input_dim]
        
        # Weighted combination
        weighted_output = (selected_outputs * top_k_weights.unsqueeze(-1)).sum(dim=2)
        
        # Output projection
        output = self.output_projection(weighted_output)
        
        return output


class EntityFlowTransformer(nn.Module):
    """Entity Flow: Models entity-specific temporal patterns."""
    
    def __init__(self, 
                 feature_dim: int = GRAPH_HIDDEN_DIM,
                 num_heads: int = GRAPH_ATTENTION_HEADS,
                 num_layers: int = 3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Entity-specific temporal attention
        self.entity_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 4, feature_dim)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers * 2)
        ])
        
    def forward(self, 
                entity_features: torch.Tensor,
                entity_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply entity flow transformer.
        
        Args:
            entity_features: [batch_size, num_entities, seq_len, feature_dim]
            entity_mask: [batch_size, num_entities, seq_len] mask for valid entities
            
        Returns:
            Updated entity features
        """
        batch_size, num_entities, seq_len, feature_dim = entity_features.shape
        
        # Process each entity separately
        updated_features = []
        for entity_idx in range(num_entities):
            entity_seq = entity_features[:, entity_idx, :, :]  # [batch_size, seq_len, feature_dim]
            entity_seq_mask = entity_mask[:, entity_idx, :]  # [batch_size, seq_len]
            
            # Apply transformer layers
            for layer_idx in range(len(self.entity_attention_layers)):
                # Self-attention
                attn_output, _ = self.entity_attention_layers[layer_idx](
                    entity_seq, entity_seq, entity_seq,
                    key_padding_mask=~entity_seq_mask
                )
                entity_seq = self.layer_norms[layer_idx * 2](entity_seq + attn_output)
                
                # Feed-forward
                ffn_output = self.ffn_layers[layer_idx](entity_seq)
                entity_seq = self.layer_norms[layer_idx * 2 + 1](entity_seq + ffn_output)
            
            updated_features.append(entity_seq)
        
        return torch.stack(updated_features, dim=1)  # [batch_size, num_entities, seq_len, feature_dim]


class TimeFlowTransformer(nn.Module):
    """Time Flow: Models temporal evolution across all entities."""
    
    def __init__(self, 
                 feature_dim: int = GRAPH_HIDDEN_DIM,
                 num_heads: int = GRAPH_ATTENTION_HEADS,
                 num_layers: int = 3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Temporal attention layers
        self.temporal_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 4, feature_dim)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers * 2)
        ])
        
    def forward(self, 
                temporal_features: torch.Tensor,
                temporal_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply time flow transformer.
        
        Args:
            temporal_features: [batch_size, seq_len, num_entities, feature_dim]
            temporal_mask: [batch_size, seq_len, num_entities] mask for valid entities
            
        Returns:
            Updated temporal features
        """
        batch_size, seq_len, num_entities, feature_dim = temporal_features.shape
        
        # Process each time step separately
        updated_features = []
        for time_idx in range(seq_len):
            time_slice = temporal_features[:, time_idx, :, :]  # [batch_size, num_entities, feature_dim]
            time_mask = temporal_mask[:, time_idx, :]  # [batch_size, num_entities]
            
            # Apply transformer layers
            for layer_idx in range(len(self.temporal_attention_layers)):
                # Self-attention
                attn_output, _ = self.temporal_attention_layers[layer_idx](
                    time_slice, time_slice, time_slice,
                    key_padding_mask=~time_mask
                )
                time_slice = self.layer_norms[layer_idx * 2](time_slice + attn_output)
                
                # Feed-forward
                ffn_output = self.ffn_layers[layer_idx](time_slice)
                time_slice = self.layer_norms[layer_idx * 2 + 1](time_slice + ffn_output)
            
            updated_features.append(time_slice)
        
        return torch.stack(updated_features, dim=1)  # [batch_size, seq_len, num_entities, feature_dim]


class CrossFlowAttention(nn.Module):
    """Cross-Flow Attention: Interaction between different flows."""
    
    def __init__(self, 
                 feature_dim: int = GRAPH_HIDDEN_DIM,
                 num_heads: int = GRAPH_ATTENTION_HEADS):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Cross-flow attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Flow fusion
        self.flow_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, 
                entity_flow: torch.Tensor,
                time_flow: torch.Tensor,
                entity_mask: torch.Tensor,
                temporal_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-flow attention.
        
        Args:
            entity_flow: [batch_size, num_entities, seq_len, feature_dim]
            time_flow: [batch_size, seq_len, num_entities, feature_dim]
            entity_mask: [batch_size, num_entities, seq_len]
            temporal_mask: [batch_size, seq_len, num_entities]
            
        Returns:
            Fused features
        """
        batch_size, num_entities, seq_len, feature_dim = entity_flow.shape
        
        # Reshape for cross-attention
        entity_flow_reshaped = entity_flow.transpose(1, 2)  # [batch_size, seq_len, num_entities, feature_dim]
        entity_flow_flat = entity_flow_reshaped.reshape(batch_size * seq_len, num_entities, feature_dim)
        entity_mask_flat = entity_mask.transpose(1, 2).reshape(batch_size * seq_len, num_entities)
        
        time_flow_flat = time_flow.reshape(batch_size * seq_len, num_entities, feature_dim)
        temporal_mask_flat = temporal_mask.reshape(batch_size * seq_len, num_entities)
        
        # Cross-attention between entity flow and time flow
        cross_output, _ = self.cross_attention(
            entity_flow_flat, time_flow_flat, time_flow_flat,
            key_padding_mask=~temporal_mask_flat
        )
        
        # Fuse flows
        fused_features = self.flow_fusion(
            torch.cat([entity_flow_flat, cross_output], dim=-1)
        )
        
        # Apply layer norm
        fused_features = self.layer_norm(fused_features)
        
        # Reshape back
        fused_features = fused_features.reshape(batch_size, seq_len, num_entities, feature_dim)
        fused_features = fused_features.transpose(1, 2)  # [batch_size, num_entities, seq_len, feature_dim]
        
        return fused_features


class HierarchicalPathReasoning(nn.Module):
    """Hierarchical Path Reasoning (HPCR) for multi-hop inference."""
    
    def __init__(self, 
                 feature_dim: int = GRAPH_HIDDEN_DIM,
                 max_path_length: int = 4,
                 num_path_types: int = 4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_path_length = max_path_length
        self.num_path_types = num_path_types
        
        # Path type embeddings
        self.path_type_embeddings = nn.Embedding(num_path_types, feature_dim)
        
        # Path reasoning networks
        self.path_reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(max_path_length - 1)
        ])
        
        # Path scoring
        self.path_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Path aggregation
        self.path_aggregator = nn.Sequential(
            nn.Linear(feature_dim * max_path_length, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, 
                node_features: torch.Tensor,
                adjacency_matrix: torch.Tensor,
                edge_types: torch.Tensor) -> Tuple[torch.Tensor, List[ReasoningPath]]:
        """
        Apply hierarchical path reasoning.
        
        Args:
            node_features: [num_nodes, feature_dim]
            adjacency_matrix: [num_nodes, num_nodes]
            edge_types: [num_edges] edge type indices
            
        Returns:
            Updated features and reasoning paths
        """
        num_nodes = node_features.shape[0]
        
        # Generate reasoning paths
        reasoning_paths = self._generate_reasoning_paths(
            node_features, adjacency_matrix, edge_types
        )
        
        # Process each path
        path_features = []
        for path in reasoning_paths:
            path_feat = self._process_reasoning_path(path, node_features)
            path_features.append(path_feat)
        
        if path_features:
            path_features = torch.stack(path_features)
            
            # Aggregate path features
            aggregated_features = self.path_aggregator(path_features.flatten())
            
            # Update node features
            updated_features = node_features + aggregated_features.unsqueeze(0).expand(num_nodes, -1)
        else:
            updated_features = node_features
        
        return updated_features, reasoning_paths
    
    def _generate_reasoning_paths(self, 
                                 node_features: torch.Tensor,
                                 adjacency_matrix: torch.Tensor,
                                 edge_types: torch.Tensor) -> List[ReasoningPath]:
        """Generate reasoning paths for multi-hop inference."""
        reasoning_paths = []
        path_id = 0
        
        num_nodes = node_features.shape[0]
        
        # Generate paths of different lengths
        for path_length in range(2, self.max_path_length + 1):
            for start_node in range(num_nodes):
                paths = self._find_paths_of_length(
                    start_node, path_length, adjacency_matrix
                )
                
                for path_nodes in paths:
                    # Determine path type based on edge types
                    path_type = self._determine_path_type(path_nodes, edge_types)
                    
                    # Create reasoning path
                    reasoning_path = ReasoningPath(
                        path_id=path_id,
                        nodes=path_nodes,
                        path_type=path_type,
                        confidence=1.0,  # Will be updated
                        features=torch.zeros(self.feature_dim)
                    )
                    
                    reasoning_paths.append(reasoning_path)
                    path_id += 1
        
        return reasoning_paths
    
    def _find_paths_of_length(self, 
                             start_node: int,
                             path_length: int,
                             adjacency_matrix: torch.Tensor) -> List[List[int]]:
        """Find all paths of given length starting from a node."""
        paths = []
        
        def dfs(current_node: int, current_path: List[int], remaining_length: int):
            if remaining_length == 0:
                paths.append(current_path[:])
                return
            
            for next_node in range(adjacency_matrix.shape[0]):
                if adjacency_matrix[current_node, next_node] > 0:
                    dfs(next_node, current_path + [next_node], remaining_length - 1)
        
        dfs(start_node, [start_node], path_length - 1)
        return paths
    
    def _determine_path_type(self, path_nodes: List[int], edge_types: torch.Tensor) -> str:
        """Determine the type of reasoning path."""
        # Simplified path type determination
        # In practice, this would analyze the edge types along the path
        
        path_types = ["causal", "spatial", "temporal", "social"]
        return path_types[len(path_nodes) % len(path_types)]
    
    def _process_reasoning_path(self, 
                               path: ReasoningPath,
                               node_features: torch.Tensor) -> torch.Tensor:
        """Process a reasoning path to extract features."""
        path_features = []
        
        for i in range(len(path.nodes) - 1):
            current_node = path.nodes[i]
            next_node = path.nodes[i + 1]
            
            # Get node features
            current_feat = node_features[current_node]
            next_feat = node_features[next_node]
            
            # Process step
            if i < len(self.path_reasoning_layers):
                step_feat = self.path_reasoning_layers[i](
                    torch.cat([current_feat, next_feat], dim=-1)
                )
            else:
                step_feat = (current_feat + next_feat) / 2
            
            path_features.append(step_feat)
        
        # Concatenate all step features
        if path_features:
            path_feat = torch.cat(path_features, dim=-1)
        else:
            path_feat = torch.zeros(self.feature_dim * self.max_path_length)
        
        # Score the path
        path_score = torch.sigmoid(self.path_scorer(path_feat[:self.feature_dim]))
        path.confidence = path_score.item()
        path.features = path_feat
        
        return path_feat


class GTRFormer(nn.Module):
    """Graph-Temporal Reasoning Transformer (GTR-Former)."""
    
    def __init__(self, 
                 feature_dim: int = GRAPH_HIDDEN_DIM,
                 num_heads: int = GRAPH_ATTENTION_HEADS,
                 num_layers: int = 3,
                 num_experts: int = 8,
                 max_path_length: int = 4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Entity Flow Transformer
        self.entity_flow = EntityFlowTransformer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Time Flow Transformer
        self.time_flow = TimeFlowTransformer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Cross-Flow Attention
        self.cross_flow_attention = CrossFlowAttention(
            feature_dim=feature_dim,
            num_heads=num_heads
        )
        
        # Mixture-of-Experts
        self.moe_layers = nn.ModuleList([
            MixtureOfExperts(
                input_dim=feature_dim,
                expert_dim=feature_dim * 2,
                num_experts=num_experts
            ) for _ in range(num_layers)
        ])
        
        # Hierarchical Path Reasoning
        self.hierarchical_reasoning = HierarchicalPathReasoning(
            feature_dim=feature_dim,
            max_path_length=max_path_length
        )
        
        # Output projection
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, 
                node_features: torch.Tensor,
                adjacency_matrix: torch.Tensor,
                edge_features: torch.Tensor,
                edge_types: torch.Tensor,
                temporal_info: Optional[torch.Tensor] = None) -> Dict:
        """
        Apply GTR-Former reasoning.
        
        Args:
            node_features: [num_nodes, feature_dim]
            adjacency_matrix: [num_nodes, num_nodes]
            edge_features: [num_edges, feature_dim]
            edge_types: [num_edges] edge type indices
            temporal_info: Optional temporal information
            
        Returns:
            Dictionary containing updated features and reasoning information
        """
        num_nodes = node_features.shape[0]
        
        # Prepare entity flow input
        if temporal_info is not None:
            # Reshape for entity flow: [batch_size, num_entities, seq_len, feature_dim]
            entity_flow_input = temporal_info.unsqueeze(0)  # Add batch dimension
            entity_mask = torch.ones(entity_flow_input.shape[:3], dtype=torch.bool)
        else:
            # Use node features as temporal sequence
            entity_flow_input = node_features.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, feature_dim]
            entity_mask = torch.ones(1, 1, num_nodes, dtype=torch.bool)
        
        # Apply Entity Flow
        entity_flow_output = self.entity_flow(entity_flow_input, entity_mask)
        
        # Prepare time flow input
        if temporal_info is not None:
            time_flow_input = temporal_info.unsqueeze(0)  # Add batch dimension
            temporal_mask = torch.ones(time_flow_input.shape[:3], dtype=torch.bool)
        else:
            # Use node features as temporal sequence
            time_flow_input = node_features.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, feature_dim]
            temporal_mask = torch.ones(1, 1, num_nodes, dtype=torch.bool)
        
        # Apply Time Flow
        time_flow_output = self.time_flow(time_flow_input, temporal_mask)
        
        # Apply Cross-Flow Attention
        cross_flow_output = self.cross_flow_attention(
            entity_flow_output, time_flow_output, entity_mask, temporal_mask
        )
        
        # Extract features for further processing
        if temporal_info is not None:
            processed_features = cross_flow_output.squeeze(0).mean(dim=1)  # [num_entities, feature_dim]
        else:
            processed_features = cross_flow_output.squeeze(0).squeeze(0)  # [num_nodes, feature_dim]
        
        # Apply Mixture-of-Experts layers
        for moe_layer in self.moe_layers:
            moe_input = processed_features.unsqueeze(0)  # Add batch dimension
            moe_output = moe_layer(moe_input)
            processed_features = moe_output.squeeze(0)
        
        # Apply Hierarchical Path Reasoning
        reasoning_features, reasoning_paths = self.hierarchical_reasoning(
            processed_features, adjacency_matrix, edge_types
        )
        
        # Output projection
        output_features = self.output_projection(reasoning_features)
        output_features = self.layer_norm(output_features)
        
        return {
            'node_features': output_features,
            'reasoning_paths': reasoning_paths,
            'entity_flow_output': entity_flow_output,
            'time_flow_output': time_flow_output,
            'cross_flow_output': cross_flow_output
        }
    
    def get_reasoning_paths(self, reasoning_paths: List[ReasoningPath]) -> Dict[str, List[ReasoningPath]]:
        """Organize reasoning paths by type."""
        paths_by_type = {
            'causal': [],
            'spatial': [],
            'temporal': [],
            'social': []
        }
        
        for path in reasoning_paths:
            if path.path_type in paths_by_type:
                paths_by_type[path.path_type].append(path)
        
        return paths_by_type
    
    def analyze_reasoning_patterns(self, reasoning_paths: List[ReasoningPath]) -> Dict:
        """Analyze reasoning patterns and extract insights."""
        if not reasoning_paths:
            return {}
        
        # Analyze path types
        path_type_counts = {}
        path_lengths = []
        path_confidences = []
        
        for path in reasoning_paths:
            path_type_counts[path.path_type] = path_type_counts.get(path.path_type, 0) + 1
            path_lengths.append(len(path.nodes))
            path_confidences.append(path.confidence)
        
        # Calculate statistics
        analysis = {
            'total_paths': len(reasoning_paths),
            'path_type_distribution': path_type_counts,
            'average_path_length': np.mean(path_lengths),
            'average_confidence': np.mean(path_confidences),
            'high_confidence_paths': len([p for p in reasoning_paths if p.confidence > 0.8]),
            'most_common_path_type': max(path_type_counts.items(), key=lambda x: x[1])[0] if path_type_counts else None
        }
        
        return analysis
