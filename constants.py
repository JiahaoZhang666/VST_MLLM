CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 8
MAX_FRAMES = 32
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SIZE = 1024
AUDIO_HOP_SIZE = 512
AUDIO_N_MELS = 128
AUDIO_N_FFT = 2048

# Spatiotemporal Graph Constants
MAX_ENTITIES = 50
MAX_TEMPORAL_EDGES = 100
SPATIAL_DISTANCE_THRESHOLD = 0.1
TEMPORAL_WINDOW_SIZE = 16
GRAPH_ATTENTION_HEADS = 8
GRAPH_HIDDEN_DIM = 512

# Narrative Flow Constants
MAX_EVENTS = 100
CAUSAL_TEMPORAL_THRESHOLD = 0.8
NARRATIVE_COHERENCE_THRESHOLD = 0.7
STORYLINE_MAX_LENGTH = 20

# Audio-Visual Disentanglement Constants
DISENTANGLEMENT_TEMPERATURE = 0.1
ENTITY_SPECIFIC_DIM = 256
EVENT_SPECIFIC_DIM = 256
CONTEXT_SPECIFIC_DIM = 256

# Multi-scale Processing Constants
TEMPORAL_SCALES = [1, 2, 4, 8, 16]  # Frame sampling rates
SPATIAL_SCALES = [1, 2, 4, 8]        # Spatial downsampling rates

MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
    "<audio>": -202,
}

# Enhanced modality support
MULTIMODAL_TOKENS = {
    "video_audio": "<video><audio>",
    "image_audio": "<image><audio>",
    "video_image": "<video><image>",
}

# Spatiotemporal relationship types
RELATIONSHIP_TYPES = [
    "spatial_proximity",
    "temporal_sequence", 
    "causal_influence",
    "interaction_pattern",
    "narrative_connection"
]

# Event categories for narrative understanding
EVENT_CATEGORIES = [
    "entity_movement",
    "interaction_start",
    "interaction_end",
    "state_change",
    "causal_trigger",
    "narrative_development"
]
