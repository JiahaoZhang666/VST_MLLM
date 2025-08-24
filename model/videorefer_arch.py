# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn

from .projector import load_mm_projector, build_vision_projector
from .encoder import build_vision_tower
from ..constants import IGNORE_INDEX, NUM_FRAMES, MODAL_INDEX_MAP
from .layer import build_region_encoder
from ..audio_processor import AudioProcessor, MultiScaleAudioProcessor, AudioVisualDisentangler
from ..spatiotemporal_graph import SpatiotemporalGraphBuilder, EntityNode
from ..narrative_flow import NarrativeFlowPredictor
from ..entity_extraction import SpatioTemporalEntityExtractor
from ..actor_graph import SpatioTemporalActorGraph
from ..gtr_former import GTRFormer
from ..task_heads import TaskSpecificHeads


class VideoSpatioTemporalMetaModel:

    def __init__(self, config):
        super(VideoSpatioTemporalMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.region_encoder = build_region_encoder(config, config.image_aspect_ratio) 
            
            # Initialize VideoSpatioTemporal components
            self.audio_processor = AudioProcessor(
                sample_rate=getattr(config, 'audio_sample_rate', 16000),
                hidden_dim=getattr(config, 'audio_hidden_dim', 512)
            )
            
            self.multi_scale_audio_processor = MultiScaleAudioProcessor(
                base_processor=self.audio_processor,
                temporal_scales=getattr(config, 'temporal_scales', [1, 2, 4, 8, 16])
            )
            
            self.audio_visual_disentangler = AudioVisualDisentangler(
                feature_dim=getattr(config, 'disentanglement_dim', 512)
            )
            
            self.spatiotemporal_graph_builder = SpatiotemporalGraphBuilder(
                feature_dim=getattr(config, 'graph_feature_dim', 512),
                max_entities=getattr(config, 'max_entities', 50),
                hidden_dim=getattr(config, 'graph_hidden_dim', 512)
            )
            
            self.narrative_flow_predictor = NarrativeFlowPredictor(
                feature_dim=getattr(config, 'narrative_feature_dim', 512),
                max_events=getattr(config, 'max_events', 100),
                hidden_dim=getattr(config, 'narrative_hidden_dim', 256)
            )
            
            # New VideoSpatioTemporal components
            self.entity_extractor = SpatioTemporalEntityExtractor(
                max_entities=getattr(config, 'max_entities', 50),
                feature_dim=getattr(config, 'entity_feature_dim', 512)
            )
            
            self.actor_graph = SpatioTemporalActorGraph(
                feature_dim=getattr(config, 'actor_graph_feature_dim', 512),
                max_entities=getattr(config, 'max_entities', 50),
                temporal_window=getattr(config, 'temporal_window', 16),
                num_attention_heads=getattr(config, 'actor_graph_heads', 8)
            )
            
            self.gtr_former = GTRFormer(
                feature_dim=getattr(config, 'gtr_feature_dim', 512),
                num_heads=getattr(config, 'gtr_heads', 8),
                num_layers=getattr(config, 'gtr_layers', 3),
                num_experts=getattr(config, 'gtr_experts', 8),
                max_path_length=getattr(config, 'max_path_length', 4)
            )
            
            self.task_heads = TaskSpecificHeads(
                feature_dim=getattr(config, 'task_feature_dim', 512),
                hidden_dim=getattr(config, 'task_hidden_dim', 256)
            )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_processor(self):
        return self.audio_processor

    def get_spatiotemporal_graph_builder(self):
        return self.spatiotemporal_graph_builder

    def get_narrative_flow_predictor(self):
        return self.narrative_flow_predictor

    def get_entity_extractor(self):
        return self.entity_extractor

    def get_actor_graph(self):
        return self.actor_graph

    def get_gtr_former(self):
        return self.gtr_former

    def get_task_heads(self):
        return self.task_heads

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_region_encoder = model_args.pretrain_region_encoder

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_region_encoder_type = getattr(model_args, 'mm_region_encoder_type', 'onefusion')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'region_encoder', None) is None:
            self.region_encoder = build_region_encoder(self.config, 'square') ## FIXME: self.config.image_aspect_ratio
        else:
            # In case it is frozen by LoRA
            for p in self.region_encoder.parameters():
                p.requires_grad = True

        # Initialize VideoSpatioTemporal components if not already done
        if not hasattr(self, 'audio_processor'):
            self.audio_processor = AudioProcessor(
                sample_rate=getattr(model_args, 'audio_sample_rate', 16000),
                hidden_dim=getattr(model_args, 'audio_hidden_dim', 512)
            )
            
        if not hasattr(self, 'spatiotemporal_graph_builder'):
            self.spatiotemporal_graph_builder = SpatiotemporalGraphBuilder(
                feature_dim=getattr(model_args, 'graph_feature_dim', 512),
                max_entities=getattr(model_args, 'max_entities', 50),
                hidden_dim=getattr(model_args, 'graph_hidden_dim', 512)
            )
            
        if not hasattr(self, 'narrative_flow_predictor'):
            self.narrative_flow_predictor = NarrativeFlowPredictor(
                feature_dim=getattr(model_args, 'narrative_feature_dim', 512),
                max_events=getattr(model_args, 'max_events', 100),
                hidden_dim=getattr(model_args, 'narrative_hidden_dim', 256)
            )
            
        # Initialize new components
        if not hasattr(self, 'entity_extractor'):
            self.entity_extractor = SpatioTemporalEntityExtractor(
                max_entities=getattr(model_args, 'max_entities', 50),
                feature_dim=getattr(model_args, 'entity_feature_dim', 512)
            )
            
        if not hasattr(self, 'actor_graph'):
            self.actor_graph = SpatioTemporalActorGraph(
                feature_dim=getattr(model_args, 'actor_graph_feature_dim', 512),
                max_entities=getattr(model_args, 'max_entities', 50),
                temporal_window=getattr(model_args, 'temporal_window', 16),
                num_attention_heads=getattr(model_args, 'actor_graph_heads', 8)
            )
            
        if not hasattr(self, 'gtr_former'):
            self.gtr_former = GTRFormer(
                feature_dim=getattr(model_args, 'gtr_feature_dim', 512),
                num_heads=getattr(model_args, 'gtr_heads', 8),
                num_layers=getattr(model_args, 'gtr_layers', 3),
                num_experts=getattr(model_args, 'gtr_experts', 8),
                max_path_length=getattr(model_args, 'max_path_length', 4)
            )
            
        if not hasattr(self, 'task_heads'):
            self.task_heads = TaskSpecificHeads(
                feature_dim=getattr(model_args, 'task_feature_dim', 512),
                hidden_dim=getattr(model_args, 'task_hidden_dim', 256)
            )
    
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
        
        if pretrain_region_encoder is not None:
            region_encoder_weights = torch.load(pretrain_region_encoder, map_location='cpu')
            self.region_encoder.load_state_dict(get_w(region_encoder_weights, 'region_encoder'))

    def build_spatiotemporal_graph(self, video_features, masks, ann_indices, frame_nums, audio=None):
        """
        Build spatiotemporal graph from video features and entity detections.
        
        Args:
            video_features: Video frame features
            masks: Object masks
            ann_indices: Annotation indices
            frame_nums: Number of frames
            audio: Optional audio features
            
        Returns:
            Spatiotemporal graph representation
        """
        # Convert masks and annotations to entity detections
        entity_detections = self._convert_masks_to_entities(masks, ann_indices, frame_nums, video_features)
        
        # Build spatiotemporal graph
        spatiotemporal_graph = self.spatiotemporal_graph_builder(
            video_features=video_features,
            entity_detections=entity_detections,
            masks=masks,
            audio_features=audio
        )
        
        return spatiotemporal_graph

    def _convert_masks_to_entities(self, masks, ann_indices, frame_nums, video_features):
        """
        Convert mask annotations to entity detections for graph construction.
        
        Args:
            masks: Object masks
            ann_indices: Annotation indices
            frame_nums: Number of frames
            video_features: Video features
            
        Returns:
            List of entity detections per frame
        """
        entity_detections = []
        
        if masks is None or len(masks) == 0:
            return entity_detections
            
        # Process each frame
        current_frame = 0
        for frame_idx, frame_num in enumerate(frame_nums):
            frame_entities = []
            
            # Get entities for this frame
            if frame_idx < len(ann_indices):
                frame_ann_indices = ann_indices[frame_idx]
                
                for ann_idx, ann_index in enumerate(frame_ann_indices):
                    if ann_idx < len(masks):
                        mask = masks[ann_idx]
                        
                        # Calculate mask center as entity position
                        if mask.sum() > 0:
                            # Find mask center
                            y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
                            center_y = y_coords.float().mean()
                            center_x = x_coords.float().mean()
                            
                            # Create entity node
                            entity = EntityNode(
                                entity_id=len(frame_entities),
                                frame_id=current_frame,
                                position=torch.tensor([center_x, center_y], device=masks.device),
                                features=torch.randn(512, device=masks.device),  # Placeholder features
                                confidence=0.8,  # Placeholder confidence
                                entity_type="object"
                            )
                            
                            frame_entities.append(entity)
            
            entity_detections.append(frame_entities)
            current_frame += frame_num
            
        return entity_detections

    def extract_entity_trajectories(self, spatiotemporal_graph):
        """
        Extract entity trajectories from spatiotemporal graph.
        
        Args:
            spatiotemporal_graph: Spatiotemporal graph
            
        Returns:
            Dictionary of entity trajectories
        """
        trajectories = {}
        
        if 'entities' not in spatiotemporal_graph:
            return trajectories
            
        # Group entities by ID across frames
        for entity in spatiotemporal_graph['entities']:
            entity_id = entity.entity_id
            
            if entity_id not in trajectories:
                trajectories[entity_id] = []
                
            trajectories[entity_id].append({
                'frame_id': entity.frame_id,
                'position': entity.position.cpu().numpy(),
                'confidence': entity.confidence,
                'entity_type': entity.entity_type
            })
        
        # Sort trajectories by frame_id
        for entity_id in trajectories:
            trajectories[entity_id] = sorted(trajectories[entity_id], key=lambda x: x['frame_id'])
            
        return trajectories

    def predict_narrative_flow(self, video_features, masks, ann_indices, frame_nums, audio=None):
        """
        Predict narrative flow from video features.
        
        Args:
            video_features: Video frame features
            masks: Object masks
            ann_indices: Annotation indices
            frame_nums: Number of frames
            audio: Optional audio features
            
        Returns:
            Narrative flow prediction results
        """
        # Build spatiotemporal graph first
        spatiotemporal_graph = self.build_spatiotemporal_graph(
            video_features, masks, ann_indices, frame_nums, audio
        )
        
        # Predict narrative flow
        narrative_results = self.narrative_flow_predictor(
            video_features=video_features,
            spatiotemporal_graph=spatiotemporal_graph,
            audio_features=audio
        )
        
        return narrative_results

    def extract_causal_relationships(self, narrative_flow):
        """
        Extract causal relationships from narrative flow.
        
        Args:
            narrative_flow: Narrative flow results
            
        Returns:
            List of causal relationships
        """
        if 'causal_relationships' in narrative_flow:
            return narrative_flow['causal_relationships']
        return []

    def disentangle_audio_visual_features(self, video_features, audio, masks, ann_indices):
        """
        Disentangle audio-visual features into entity, event, and context components.
        
        Args:
            video_features: Video frame features
            audio: Audio features
            masks: Object masks
            ann_indices: Annotation indices
            
        Returns:
            Disentangled features dictionary
        """
        if audio is None:
            return None
            
        # Process audio features
        audio_features = self.audio_processor(audio, video_features)
        
        # Aggregate video features
        if video_features.dim() > 3:
            video_features = video_features.mean(dim=0)  # Average over batch
        video_features = video_features.mean(dim=0)  # Average over time
        
        # Disentangle features
        disentangled_features = self.audio_visual_disentangler(
            audio_features=audio_features['temporal_features'],
            visual_features=video_features
        )
        
        return disentangled_features

    def analyze_spatiotemporal_relationships(self, video_features, **kwargs):
        """
        Analyze spatiotemporal relationships in video.
        
        Args:
            video_features: Video frame features
            **kwargs: Additional analysis parameters
            
        Returns:
            Spatiotemporal analysis results
        """
        # Extract basic spatiotemporal patterns
        temporal_features = video_features.mean(dim=(-2, -1))  # Average spatial dimensions
        
        # Analyze temporal patterns
        temporal_analysis = {
            'temporal_variance': torch.var(temporal_features, dim=0),
            'temporal_mean': torch.mean(temporal_features, dim=0),
            'temporal_correlation': torch.corrcoef(temporal_features.T),
            'frame_count': video_features.shape[0]
        }
        
        return temporal_analysis

    def process_video_comprehensive(self, video_features, audio=None, masks=None, ann_indices=None, frame_nums=None):
        """
        Comprehensive video processing pipeline using all VideoSpatioTemporal components.
        
        Args:
            video_features: Video frame features
            audio: Audio features
            masks: Object masks
            ann_indices: Annotation indices
            frame_nums: Number of frames
            
        Returns:
            Comprehensive analysis results
        """
        results = {}
        
        # 1. Entity Extraction
        if hasattr(self, 'entity_extractor'):
            entity_detections = self.entity_extractor(video_features, audio)
            results['entity_detections'] = entity_detections
        
        # 2. Actor Graph Construction
        if hasattr(self, 'actor_graph') and 'entity_detections' in results:
            actor_graph_results = self.actor_graph(
                entity_detections=results['entity_detections'],
                audio_features=audio
            )
            results['actor_graph'] = actor_graph_results
        
        # 3. GTR-Former Reasoning
        if hasattr(self, 'gtr_former') and 'actor_graph' in results:
            gtr_results = self.gtr_former(
                node_features=results['actor_graph']['node_features'],
                adjacency_matrix=results['actor_graph']['adjacency_matrix'],
                edge_features=results['actor_graph']['edge_features'],
                edge_types=results['actor_graph']['edge_types']
            )
            results['gtr_reasoning'] = gtr_results
        
        # 4. Task-Specific Analysis
        if hasattr(self, 'task_heads') and 'gtr_reasoning' in results:
            # VideoQA
            qa_results = self.task_heads(
                gtr_output=results['gtr_reasoning'],
                task_type='videoqa',
                question_features=torch.randn(512),  # Placeholder
                question_text="What is happening in this video?"
            )
            results['videoqa'] = qa_results
            
            # Relationship Recognition
            rel_results = self.task_heads(
                gtr_output=results['gtr_reasoning'],
                task_type='relationship',
                actor1_id=0,
                actor2_id=1
            )
            results['relationships'] = rel_results
            
            # Event Prediction
            event_results = self.task_heads(
                gtr_output=results['gtr_reasoning'],
                task_type='event_prediction',
                actor_ids=[0, 1]
            )
            results['event_prediction'] = event_results
        
        return results


class VideoSpatioTemporalMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_processor(self):
        return self.get_model().get_audio_processor()

    def get_spatiotemporal_graph_builder(self):
        return self.get_model().get_spatiotemporal_graph_builder()

    def get_narrative_flow_predictor(self):
        return self.get_model().get_narrative_flow_predictor()

    def get_entity_extractor(self):
        return self.get_model().get_entity_extractor()

    def get_actor_graph(self):
        return self.get_model().get_actor_graph()

    def get_gtr_former(self):
        return self.get_model().get_gtr_former()

    def get_task_heads(self):
        return self.get_model().get_task_heads()

    def encode_images_or_videos(self, images):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        data_batch = []
        for i, (data, modal) in enumerate(images):
            if modal == 'image':
                data = data.expand(num_frames, -1, -1, -1)
            else:
                data = data
            data_batch.append(data)

        data_batch = torch.stack(data_batch, dim=0)

        assert len(data_batch.size()) == 5
        batch_size = data_batch.size(0)

        frames = einops.rearrange(data_batch, 'b t c h w -> (b t) c h w')
        frames_features = self.get_model().get_vision_tower()(frames)
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)

        return self.temporal_aggregator(frames_features)

    def temporal_aggregator(self, frames_features):
        """Temporal aggregation of frame features.
        Args:
            frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
        Returns:
            torch.Tensor: Video features with shape (b, n, h).
        """
        # TODO: improve the merging method.
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        # *********** spatial convolution *************
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** spatial pooling *************
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")

        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, masks, frame, ann_indices, frame_nums, audio=None
    ):
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        mm_features = self.encode_images_or_videos(images)
        if frame is not None:
            first_frame_features = self.get_model().get_vision_tower()(torch.cat(frame, dim=0))
            mask_feats, region_token_nums = self.get_model().region_encoder(first_frame_features, masks, mm_features, ann_indices, frame_nums)
        else:
            mask_feats = []
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx = 0

        cur_region_idx = 0
        cur_region_pos = 0
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # pure text input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_mask_feat = mask_feats[cur_region_idx:cur_region_idx+1]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                if frame is not None:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_mask_feat[0:0], cur_input_embeds_2], dim=0)
                else:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1 
                cur_region_idx += 1
                cur_region_pos += 1
                continue

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            while mm_token_indices.numel() > 0:
                cur_mm_features = mm_features[cur_mm_idx]
                mm_token_start = mm_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start])) 
                cur_new_input_embeds.append(cur_mm_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                cur_mm_idx += 1
                cur_input_ids = cur_input_ids[mm_token_start+1:] 
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            if cur_input_ids.numel() > 0:
                region_idx = torch.nonzero(cur_input_ids==self.tokenizer.convert_tokens_to_ids(['<region>'])[0])
                if len(region_idx)==0:
                    if frame is not None:
                        cur_mask_feat = mask_feats[cur_region_idx:cur_region_idx+1]
                        cur_new_input_embeds.append(cur_mask_feat[0:0].to(cur_mm_features.dtype))
                    cur_region_idx += 1
                    cur_region_pos += 1
                _l = 0
                for idx in region_idx:
                    cur_raw_new_input_embeds = self.get_model().embed_tokens(cur_input_ids[_l:idx[0]])
                    cur_new_input_embeds.append(cur_raw_new_input_embeds)
                    if labels is not None:
                        cur_labels_ = cur_labels[_l:idx[0]]
                        cur_new_labels.append(cur_labels_)
                    ## mask
                    cur_new_input_embeds.append(mask_feats[cur_region_idx:cur_region_idx+region_token_nums[cur_region_pos]].to(cur_raw_new_input_embeds.dtype))
                    if labels is not None:
                        cur_new_labels.append(torch.full((region_token_nums[cur_region_pos],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_region_idx+=region_token_nums[cur_region_pos]
                    cur_region_pos+=1
                   
                    _l = idx[0]+1

                if _l< len(cur_input_ids):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[_l:]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[_l:])

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # if cur_region_idx!=len(mask_feats):
        #     print(f'cur region idx {cur_region_idx} not equal to mask feats{len(mask_feats)}')

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_MM_tokenizer(self, tokenizer):
  
        # add region token
        num_new_tokens = tokenizer.add_tokens('<region>', special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

       
        for m in self.modules():
            m.tokenizer = tokenizer

