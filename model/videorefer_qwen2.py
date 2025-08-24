# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videorefer_arch import VideoSpatioTemporalMetaModel, VideoSpatioTemporalMetaForCausalLM


class VideoSpatioTemporalQwen2Config(Qwen2Config):
    model_type = "videospatiotemporal_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videospatiotemporal_qwen2"
        
        # VideoSpatioTemporal specific configurations
        self.audio_sample_rate = getattr(kwargs, 'audio_sample_rate', 16000)
        self.audio_hidden_dim = getattr(kwargs, 'audio_hidden_dim', 512)
        self.graph_feature_dim = getattr(kwargs, 'graph_feature_dim', 512)
        self.max_entities = getattr(kwargs, 'max_entities', 50)
        self.graph_hidden_dim = getattr(kwargs, 'graph_hidden_dim', 512)
        self.narrative_feature_dim = getattr(kwargs, 'narrative_feature_dim', 512)
        self.max_events = getattr(kwargs, 'max_events', 100)
        self.narrative_hidden_dim = getattr(kwargs, 'narrative_hidden_dim', 256)
        self.disentanglement_dim = getattr(kwargs, 'disentanglement_dim', 512)
        self.temporal_scales = getattr(kwargs, 'temporal_scales', [1, 2, 4, 8, 16])


class VideoSpatioTemporalQwen2Model(VideoSpatioTemporalMetaModel, Qwen2Model):
    config_class = VideoSpatioTemporalQwen2Config

    def __init__(self, config: VideoSpatioTemporalQwen2Config):
        super(VideoSpatioTemporalQwen2Model, self).__init__(config)


class VideoSpatioTemporalQwen2ForCausalLM(Qwen2ForCausalLM, VideoSpatioTemporalMetaForCausalLM):
    config_class = VideoSpatioTemporalQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VideoSpatioTemporalQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        masks = None,
        frame = None,
        ann_indices = None,
        frame_nums = None,
        audio = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                masks,
                frame,
                ann_indices,
                frame_nums,
                audio,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        masks = None,
        frame = None,
        ann_indices = None,
        frame_nums = None,
        audio = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
                masks=masks,
                frame=frame,
                ann_indices=ann_indices,
                frame_nums=frame_nums,
                audio=audio,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        audio = kwargs.pop("audio", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds
        )
        if images is not None:
            _inputs['images'] = images
        if audio is not None:
            _inputs['audio'] = audio
        return _inputs

    def build_spatiotemporal_graph(self, video_features, masks, ann_indices, frame_nums, audio=None):
        """Build spatiotemporal graph from video features."""
        return self.get_model().build_spatiotemporal_graph(
            video_features, masks, ann_indices, frame_nums, audio
        )

    def extract_entity_trajectories(self, spatiotemporal_graph):
        """Extract entity trajectories from spatiotemporal graph."""
        return self.get_model().extract_entity_trajectories(spatiotemporal_graph)

    def predict_narrative_flow(self, video_features, masks, ann_indices, frame_nums, audio=None):
        """Predict narrative flow from video features."""
        return self.get_model().predict_narrative_flow(
            video_features, masks, ann_indices, frame_nums, audio
        )

    def extract_causal_relationships(self, narrative_flow):
        """Extract causal relationships from narrative flow."""
        return self.get_model().extract_causal_relationships(narrative_flow)

    def disentangle_audio_visual_features(self, video_features, audio, masks, ann_indices):
        """Disentangle audio-visual features."""
        return self.get_model().disentangle_audio_visual_features(
            video_features, audio, masks, ann_indices
        )

    def analyze_spatiotemporal_relationships(self, video_features, **kwargs):
        """Analyze spatiotemporal relationships in video."""
        return self.get_model().analyze_spatiotemporal_relationships(video_features, **kwargs)


AutoConfig.register("videospatiotemporal_qwen2", VideoSpatioTemporalQwen2Config)
AutoModelForCausalLM.register(VideoSpatioTemporalQwen2Config, VideoSpatioTemporalQwen2ForCausalLM)
