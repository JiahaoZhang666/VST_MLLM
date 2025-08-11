import os
import copy
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN
from .audio_processor import AudioProcessor
from .spatiotemporal_graph import SpatiotemporalGraphBuilder
from .narrative_flow import NarrativeFlowPredictor


def model_init(model_path=None, region=None, **kwargs):
    """
    Initialize the VideoSpatioTemporal model with enhanced spatiotemporal understanding capabilities.
    
    Args:
        model_path: Path to the pretrained model
        region: Region configuration for object detection
        **kwargs: Additional model configuration parameters
    
    Returns:
        model: VideoSpatioTemporal model
        processor: Multi-modal processor
        tokenizer: Text tokenizer
    """
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, modal='video', masks=None, ann_indices=None, frame_nums=None, frame=None, audio=None, **kwargs):
    """Enhanced inference API of VideoSpatioTemporal for comprehensive video understanding.

    Args:
        model: VideoSpatioTemporal model with spatiotemporal capabilities.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        modal (str): inference modality ('image', 'video', 'audio', 'multimodal').
        masks: Object masks for region understanding
        ann_indices: Annotation indices
        frame_nums: Number of frames
        frame: Specific frame information
        audio: Audio input for multimodal understanding
        **kwargs: Additional parameters including:
            - do_sample: whether to sample
            - temperature: sampling temperature
            - top_p: top-p sampling parameter
            - max_new_tokens: maximum new tokens to generate
            - enable_spatiotemporal: enable spatiotemporal graph construction
            - enable_narrative: enable narrative flow prediction
            - enable_audio_visual: enable audio-visual disentanglement
    
    Returns:
        dict: Comprehensive understanding results including:
            - response: text response
            - spatiotemporal_graph: constructed spatiotemporal graph
            - narrative_flow: predicted narrative structure
            - entity_trajectories: tracked entity paths
            - causal_relationships: detected causal connections
    """
    
    # Enable advanced features based on configuration
    enable_spatiotemporal = kwargs.get('enable_spatiotemporal', True)
    enable_narrative = kwargs.get('enable_narrative', True)
    enable_audio_visual = kwargs.get('enable_audio_visual', True)

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'multimodal':
        modal_token = DEFAULT_VIDEO_TOKEN + DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        tensor = image_or_video.half().cuda()
        tensor = [(tensor, modal)]

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videospatiotemporal', 'videospatiotemporal_mistral', 'videospatiotemporal_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant specialized in VideoSpatioTemporal understanding. 
            You excel at analyzing spatiotemporal relationships, narrative structures, and causal connections in videos. 
            Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, 
            racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased 
            and positive in nature. Focus on providing comprehensive analysis of spatial configurations, temporal evolution, 
            and narrative coherence in the visual content.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)
    
    if frame is not None:
        frame=[frame.half().cuda()]

    # Initialize advanced understanding components
    results = {}
    
    with torch.inference_mode():
        # Generate base response
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            masks=masks,
            ann_indices=ann_indices, 
            frame_nums=frame_nums,
            frame=frame,
            audio=audio,
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        results['response'] = outputs

        # Enhanced spatiotemporal understanding
        if enable_spatiotemporal and modal in ['video', 'multimodal']:
            spatiotemporal_graph = model.build_spatiotemporal_graph(
                tensor, masks, ann_indices, frame_nums, audio
            )
            results['spatiotemporal_graph'] = spatiotemporal_graph
            
            # Extract entity trajectories
            entity_trajectories = model.extract_entity_trajectories(spatiotemporal_graph)
            results['entity_trajectories'] = entity_trajectories

        # Narrative flow prediction
        if enable_narrative and modal in ['video', 'multimodal']:
            narrative_flow = model.predict_narrative_flow(
                tensor, masks, ann_indices, frame_nums, audio
            )
            results['narrative_flow'] = narrative_flow
            
            # Extract causal relationships
            causal_relationships = model.extract_causal_relationships(narrative_flow)
            results['causal_relationships'] = causal_relationships

        # Audio-visual disentanglement
        if enable_audio_visual and audio is not None:
            disentangled_features = model.disentangle_audio_visual_features(
                tensor, audio, masks, ann_indices
            )
            results['disentangled_features'] = disentangled_features

    return results


def analyze_spatiotemporal_relationships(video, model, tokenizer, **kwargs):
    """
    Specialized function for analyzing spatiotemporal relationships in videos.
    
    Args:
        video: Video input tensor
        model: VideoSpatioTemporal model
        tokenizer: Text tokenizer
        **kwargs: Additional analysis parameters
    
    Returns:
        dict: Comprehensive spatiotemporal analysis results
    """
    return model.analyze_spatiotemporal_relationships(video, **kwargs)


def predict_narrative_flow(video, audio, model, tokenizer, **kwargs):
    """
    Specialized function for predicting narrative flow and causal relationships.
    
    Args:
        video: Video input tensor
        audio: Audio input tensor
        model: VideoSpatioTemporal model
        tokenizer: Text tokenizer
        **kwargs: Additional prediction parameters
    
    Returns:
        dict: Narrative flow prediction results
    """
    return model.predict_narrative_flow(video, audio, **kwargs)
