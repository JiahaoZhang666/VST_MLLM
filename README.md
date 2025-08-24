# VideoSpatioTemporal: A Hierarchical Cross-Modal Framework for Narrative Understanding in Videos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

VideoSpatioTemporal is a cutting-edge framework for comprehensive narrative understanding in videos. It goes beyond traditional video understanding by modeling complex spatiotemporal relationships between entities, events, and their causal connections to extract coherent storylines.

### 🎯 Key Innovations

- **Multi-scale Spatiotemporal Graph Transformer**: Dynamically models spatial relationships between entities across varying temporal scales
- **Audio-Visual Contrastive Disentanglement**: Separates entity-specific, event-specific, and context-specific features across modalities
- **Narrative Flow Prediction**: Models causal relationships between events and predicts future interactions based on narrative coherence

## 🏗️ Architecture

```
VideoSpatioTemporal Framework
├── SpatioTemporal Entity Extraction (STEE)
│   ├── Person Detection & Tracking (MOTRv2/ByteTrack)
│   ├── Behavior Modeling (Pose Estimation, Action Features)
│   ├── Speech Modeling (Whisper, Lip Sync, Spatial Audio)
│   └── Multi-modal Feature Fusion
├── SpatioTemporal Actor Graph (STAG)
│   ├── Dynamic Graph Construction
│   ├── Temporal, Spatial, and Speech Edges
│   └── Heterogeneous Graph Attention
├── Graph-Temporal Reasoning Transformer (GTR-Former)
│   ├── Entity Flow & Time Flow Modeling
│   ├── Cross-flow Attention
│   ├── Mixture-of-Experts (MoE)
│   └── Hierarchical Path Reasoning
└── Task-Specific Output Heads
    ├── Video Question Answering
    ├── Relationship Recognition
    └── Event Prediction
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from videospatiotemporal import model_init, mm_infer, analyze_spatiotemporal_relationships, predict_narrative_flow

# Initialize the model
model = model_init()

# Process a video with comprehensive analysis
results = mm_infer(
    model=model,
    video_path="path/to/video.mp4",
    audio_path="path/to/audio.wav",
    text="What is happening in this video?",
    enable_spatiotemporal=True,
    enable_narrative=True,
    enable_audio_visual=True
)

# Extract spatiotemporal relationships
spatiotemporal_analysis = analyze_spatiotemporal_relationships(
    video_features=video_features
)

# Predict narrative flow
narrative_results = predict_narrative_flow(
    video_features=video_features,
    masks=masks,
    ann_indices=ann_indices,
    frame_nums=frame_nums,
    audio=audio
)
```

### Advanced Usage

```python
# Comprehensive video processing pipeline
comprehensive_results = model.process_video_comprehensive(
    video_features=video_frames,
    audio=audio_features,
    masks=entity_masks,
    ann_indices=annotation_indices,
    frame_nums=frame_numbers
)

# Access individual components
entity_extractor = model.get_entity_extractor()
actor_graph = model.get_actor_graph()
gtr_former = model.get_gtr_former()
task_heads = model.get_task_heads()

# Build spatiotemporal graph
spatiotemporal_graph = model.build_spatiotemporal_graph(
    video_features=video_features,
    masks=masks,
    ann_indices=ann_indices,
    frame_nums=frame_nums,
    audio=audio
)

# Extract entity trajectories
trajectories = model.extract_entity_trajectories(spatiotemporal_graph)

# Extract causal relationships
causal_relationships = model.extract_causal_relationships(narrative_results)
```


## 🎓 Training

### Data Format

The framework expects data in the following format:

```json
{
  "video_id": "video_0001",
  "video_path": "videos/video_0001.mp4",
  "audio_path": "audio/video_0001.wav",
  "num_frames": 16,
  "duration": 5.2,
  "entities": [
    {
      "entity_id": 0,
      "entity_type": "person",
      "frames": [0, 1, 2, 3, 4, 5],
      "bboxes": [[x, y, w, h], ...],
      "actions": ["walking", "sitting", ...],
      "speech": [true, false, ...]
    }
  ],
  "relationships": [
    {
      "entity1_id": 0,
      "entity2_id": 1,
      "relationship_type": "friend",
      "confidence": 0.85
    }
  ],
  "events": [
    {
      "event_id": 0,
      "event_type": "conversation",
      "frame_id": 5,
      "entities_involved": [0, 1],
      "description": "Two people talking",
      "confidence": 0.92
    }
  ],
  "qa_pairs": [
    {
      "question": "What is happening in this video?",
      "answer": "Two friends are having a conversation",
      "question_type": "action"
    }
  ]
}
```
