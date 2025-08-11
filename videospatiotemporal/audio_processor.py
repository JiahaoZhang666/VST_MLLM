"""
Audio Processor Module for VideoSpatioTemporal Framework

This module handles audio processing, feature extraction, and audio-visual synchronization
for the comprehensive video understanding system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from typing import List, Tuple, Optional, Dict
from .constants import (
    AUDIO_SAMPLE_RATE, AUDIO_WINDOW_SIZE, AUDIO_HOP_SIZE, 
    AUDIO_N_MELS, AUDIO_N_FFT, MAX_ENTITIES
)


class AudioProcessor(nn.Module):
    """
    Comprehensive audio processor for multi-modal video understanding.
    
    Features:
    - Multi-scale audio feature extraction
    - Speaker diarization and identification
    - Audio-visual synchronization
    - Temporal alignment with video frames
    """
    
    def __init__(self, 
                 sample_rate: int = AUDIO_SAMPLE_RATE,
                 n_mels: int = AUDIO_N_MELS,
                 n_fft: int = AUDIO_N_FFT,
                 hop_length: int = AUDIO_HOP_SIZE,
                 win_length: int = AUDIO_WINDOW_SIZE,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.hidden_dim = hidden_dim
        
        # Mel spectrogram extraction
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            mel_scale='htk'
        )
        
        # Log-mel spectrogram
        self.log_mel = torchaudio.transforms.AmplitudeToDB()
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(n_mels, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11]
        ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Speaker embedding network
        self.speaker_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Audio-visual alignment
        self.alignment_net = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, audio: torch.Tensor, video_frames: Optional[torch.Tensor] = None) -> Dict:
        """
        Process audio input and extract comprehensive features.
        
        Args:
            audio: Audio tensor of shape (batch_size, time_steps) or (time_steps,)
            video_frames: Optional video frames for alignment
            
        Returns:
            Dictionary containing:
            - mel_features: Mel spectrogram features
            - temporal_features: Multi-scale temporal features
            - speaker_embeddings: Speaker identification features
            - aligned_features: Audio-visual aligned features
            - temporal_alignment: Temporal alignment scores
        """
        batch_size = audio.shape[0] if audio.dim() > 1 else 1
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Extract mel spectrogram
        mel_spec = self.mel_spectrogram(audio)  # (batch, n_mels, time)
        log_mel = self.log_mel(mel_spec)
        
        # Multi-scale temporal feature extraction
        temporal_features = []
        for conv in self.temporal_convs:
            feat = conv(log_mel)
            feat = F.relu(feat)
            feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # Global pooling
            temporal_features.append(feat)
        
        # Concatenate and fuse features
        temporal_features = torch.cat(temporal_features, dim=1)
        fused_features = self.feature_fusion(temporal_features)
        
        # Speaker embedding
        speaker_embeddings = self.speaker_encoder(fused_features)
        
        # Audio-visual alignment if video is provided
        aligned_features = None
        temporal_alignment = None
        
        if video_frames is not None:
            aligned_features, temporal_alignment = self._align_audio_visual(
                fused_features, video_frames
            )
        
        return {
            'mel_features': log_mel,
            'temporal_features': fused_features,
            'speaker_embeddings': speaker_embeddings,
            'aligned_features': aligned_features,
            'temporal_alignment': temporal_alignment
        }
    
    def _align_audio_visual(self, audio_features: torch.Tensor, 
                           video_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align audio and video features using cross-modal attention.
        
        Args:
            audio_features: Audio features (batch, time, hidden_dim)
            video_features: Video features (batch, time, hidden_dim)
            
        Returns:
            Tuple of (aligned_features, alignment_scores)
        """
        # Ensure compatible dimensions
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(0)
            
        # Cross-modal attention for alignment
        aligned_features, alignment_weights = self.alignment_net(
            query=audio_features,
            key=video_features,
            value=video_features
        )
        
        return aligned_features, alignment_weights
    
    def extract_speaker_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker-specific features for diarization.
        
        Args:
            audio: Audio tensor
            
        Returns:
            Speaker features
        """
        features = self.forward(audio)
        return features['speaker_embeddings']
    
    def segment_audio(self, audio: torch.Tensor, 
                     segment_length: int = 16000) -> List[torch.Tensor]:
        """
        Segment audio into fixed-length chunks for processing.
        
        Args:
            audio: Audio tensor
            segment_length: Length of each segment in samples
            
        Returns:
            List of audio segments
        """
        segments = []
        total_length = audio.shape[-1]
        
        for start in range(0, total_length, segment_length):
            end = min(start + segment_length, total_length)
            segment = audio[..., start:end]
            
            # Pad if necessary
            if segment.shape[-1] < segment_length:
                padding = segment_length - segment.shape[-1]
                segment = F.pad(segment, (0, padding))
                
            segments.append(segment)
            
        return segments
    
    def compute_audio_similarity(self, audio1: torch.Tensor, 
                               audio2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two audio segments.
        
        Args:
            audio1: First audio tensor
            audio2: Second audio tensor
            
        Returns:
            Similarity score
        """
        features1 = self.extract_speaker_features(audio1)
        features2 = self.extract_speaker_features(audio2)
        
        # Cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=-1)
        return similarity


class MultiScaleAudioProcessor(nn.Module):
    """
    Multi-scale audio processor for capturing temporal patterns at different scales.
    """
    
    def __init__(self, base_processor: AudioProcessor, 
                 temporal_scales: List[int] = None):
        super().__init__()
        
        self.base_processor = base_processor
        self.temporal_scales = temporal_scales or [1, 2, 4, 8, 16]
        
        # Scale-specific processors
        self.scale_processors = nn.ModuleList([
            nn.Conv1d(base_processor.hidden_dim, base_processor.hidden_dim, 
                      kernel_size=scale, stride=scale, padding=0)
            for scale in self.temporal_scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(base_processor.hidden_dim * len(self.temporal_scales), 
                     base_processor.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_processor.hidden_dim, base_processor.hidden_dim)
        )
    
    def forward(self, audio: torch.Tensor) -> Dict:
        """
        Extract multi-scale audio features.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Multi-scale features dictionary
        """
        # Base features
        base_features = self.base_processor(audio)
        
        # Multi-scale processing
        scale_features = []
        mel_features = base_features['mel_features']
        
        for processor in self.scale_processors:
            scale_feat = processor(mel_features)
            scale_feat = F.adaptive_avg_pool1d(scale_feat, 1).squeeze(-1)
            scale_features.append(scale_feat)
        
        # Fuse scale features
        scale_features = torch.cat(scale_features, dim=1)
        fused_scale_features = self.scale_fusion(scale_features)
        
        # Update base features
        base_features['multi_scale_features'] = fused_scale_features
        base_features['scale_features'] = scale_features
        
        return base_features


class AudioVisualDisentangler(nn.Module):
    """
    Audio-visual feature disentangler for separating entity, event, and context features.
    """
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Disentanglement networks
        self.entity_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        self.event_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        # Contrastive learning projection
        self.contrastive_projection = nn.Linear(feature_dim // 2, feature_dim // 4)
        
    def forward(self, audio_features: torch.Tensor, 
               visual_features: torch.Tensor) -> Dict:
        """
        Disentangle audio-visual features into entity, event, and context components.
        
        Args:
            audio_features: Audio features
            visual_features: Visual features
            
        Returns:
            Disentangled features dictionary
        """
        # Combine features
        combined_features = torch.cat([audio_features, visual_features], dim=-1)
        
        # Extract disentangled components
        entity_features = self.entity_encoder(combined_features)
        event_features = self.event_encoder(combined_features)
        context_features = self.context_encoder(combined_features)
        
        # Contrastive projections
        entity_proj = self.contrastive_projection(entity_features)
        event_proj = self.contrastive_projection(event_features)
        context_proj = self.contrastive_projection(context_features)
        
        return {
            'entity_features': entity_features,
            'event_features': event_features,
            'context_features': context_features,
            'entity_projection': entity_proj,
            'event_projection': event_proj,
            'context_projection': context_proj
        }
