CONST_CPU = "cpu"
CONST_GPU = "gpu"

CONST_BUFFER_CONFIG = "buffer_config"
CONST_BUFFER_PATH = "buffer_path"
CONST_CONFIG = "config"
CONST_COUNT = "count"
CONST_DATA = "data"
CONST_HYPERPARAMETERS = "hyperparameters"
CONST_HYPERPARAMS = "hyperparams"
CONST_MODEL_DICT = "model_dict"
CONST_OPTIMIZER = "optimizer"
CONST_OPT_STATE = "opt_state"
CONST_PARAMS = "params"
CONST_EVAL = "eval"
CONST_TRAIN = "train"
CONST_TEST = "test"
CONST_VALIDATION = "validation"
CONST_VAL_PREDS = "validation_predictions"
CONST_ENTROPY = "entropy"
CONST_LOG = "log"
CONST_LOGITS = "logits"
CONST_LOG_PROBS = "log_probs"
CONST_NUM_UPDATES = "num_updates"
CONST_PROBS = "probs"
CONST_RUN_PATH = "run_path"
CONST_SCALAR = "scalar"
CONST_SCALARS = "scalars"
CONST_SHAPE = "shape"
CONST_STD = "std"
CONST_VAR = "var"
CONST_DATASET_WRAPPER = "dataset_wrapper"
CONST_WRAPPER = "wrapper"
CONST_UPDATES = "updates"

VALID_SPLIT = (CONST_TRAIN, CONST_TEST)

CONST_A_MIN = "a_min"
CONST_A_MAX = "a_max"

CONST_EPS = "eps"
CONST_EPSILON = "epsilon"

CONST_REWARD = "reward"
CONST_ACTION = "action"
CONST_ADVANTAGE = "advantage"
CONST_RETURN = "return"
CONST_VALUE = "value"
CONST_ADVANTAGES = "advantages"
CONST_RETURNS = "returns"
CONST_OBSERVATION = "observation"
CONST_SATURATION = "saturation"
CONST_VALUES = "values"
CONST_LATEST_RETURN = "latest_return"
CONST_LATEST_EPISODE_LENGTH = "latest_episode_length"
CONST_AVERAGE_RETURN = "average_return"
CONST_AVERAGE_EPISODE_LENGTH = "average_episode_length"
CONST_EPISODE_LENGTHS = "episode_lengths"
CONST_EPISODIC_RETURNS = "episodic_returns"
CONST_GLOBAL_STEP = "global_step"

CONST_IS_RATIO = "is_ratio"

CONST_AUTO = "auto"
CONST_TAU = "tau"

CONST_AUX = "aux"
CONST_AGG_LOSS = "aggregate_loss"
CONST_COEFFICIENT = "coefficient"
CONST_MEAN = "mean"
CONST_METRIC = "metric"
CONST_LOSS = "loss"
CONST_PREDICTIONS = "predictions"
CONST_SUM = "sum"
CONST_ACCURACY = "accuracy"

VALID_REDUCTION = [CONST_SUM, CONST_MEAN]

CONST_OPEX = "opex"
CONST_ROBOHIVE = "robohive"

CONST_CNN = "cnn"
CONST_ENCODER_PREDICTOR = "encoder_predictor"
CONST_ENSEMBLE = "ensemble"
CONST_ICL_GPT = "icl_gpt"
CONST_GPT = "gpt"
CONST_MLP = "mlp"
CONST_RESNET = "resnet"
CONST_PATCH_EMBEDDING = "patch_embedding"
CONST_CLS_TOKEN = "cls_token"
CONST_MASK_TOKEN = "mask_token"
CONST_TEXT_EMBEDDING = "text_embedding"
CONST_TEXT = "text"
CONST_IMAGE_EMBEDDING = "image_embedding"
CONST_VISION_EMBEDDING = "vision_embedding"

CONST_BANG_BANG = "bang_bang"
CONST_DETERMINISTIC = "deterministic"
CONST_GAUSSIAN = "gaussian"
CONST_SOFTMAX = "softmax"
CONST_SQUASHED_GAUSSIAN = "squashed_gaussian"

CONST_STATE_ACTION_INPUT = "state_action_input"
VALID_Q_FUNCTION = [CONST_STATE_ACTION_INPUT]

CONST_DECODER = "decoder"
CONST_ENCODER = "encoder"
CONST_MODEL = "model"
CONST_POLICY = "policy"
CONST_CLASSIFIER = "classifier"
CONST_PREDICTOR = "predictor"
CONST_REPRESENTATION = "representation"
CONST_VF = "vf"
CONST_QF = "qf"
CONST_TARGET_QF = "target_qf"
CONST_QF_ENCODING = "qf_encoding"
CONST_TABULAR_QF = "tabular_qf"
CONST_TABULAR_VF = "tabular_vf"
CONST_TABULAR_POLICY = "tabular_policy"
CONST_QUANTIZATION = "quantization"

CONST_STD_TRANSFORM = "std_transform"

CONST_MIN_STD = "min_std"
CONST_TEMPERATURE = "temperature"

DEFAULT_MIN_STD = 1e-7
DEFAULT_TEMPERATURE = 1.0
DEFAULT_ACTOR_MIN_STD = 1e-7
DEFAULT_ACTOR_MAX_STD = 1.0

CONST_IDENTITY = "identity"
CONST_RELU = "relu"
CONST_TANH = "tanh"
VALID_ACTIVATION = [CONST_IDENTITY, CONST_RELU, CONST_TANH]

CONST_INPUT_TOKENIZER = "input_tokenizer"
CONST_OUTPUT_TOKENIZER = "output_tokenizer"

CONST_POSITIONAL_ENCODING = "positional_encoding"
CONST_NO_ENCODING = "no_encoding"
CONST_DEFAULT_ENCODING = "default"
VALID_POSITIONAL_ENCODING = [CONST_NO_ENCODING, CONST_DEFAULT_ENCODING]

CONST_CONCATENATE_INPUTS_ENCODING = "concatenate_inputs_encoding"
VALID_STATE_ACTION_ENCODING = [CONST_CONCATENATE_INPUTS_ENCODING]

VALID_Q_ENCODING = [CONST_CONCATENATE_INPUTS_ENCODING, CONST_NO_ENCODING]

CONST_SAME_PADDING = "SAME"
CONST_BATCH_STATS = "batch_stats"

CONST_ADAM = "adam"
CONST_FROZEN = "frozen"
CONST_SGD = "sgd"

VALID_OPTIMIZER = [CONST_ADAM, CONST_FROZEN, CONST_SGD]

CONST_MASK_NAMES = "mask_names"

CONST_CONSTANT_SCHEDULE = "constant_schedule"
CONST_EXPONENTIAL_DECAY = "exponential_decay"
CONST_LINEAR_SCHEDULE = "linear_schedule"
VALID_SCEHDULER = [
    CONST_CONSTANT_SCHEDULE,
    CONST_EXPONENTIAL_DECAY,
    CONST_LINEAR_SCHEDULE,
]

CONST_LEARNING_RATE = "learning_rate"

CONST_REVERSE_KL = "reverse_kl"

CONST_UPDATE_TIME = "update_time"
CONST_ROLLOUT_TIME = "rollout_time"
CONST_SAMPLING_TIME = "sampling_time"

CONST_REGULARIZATION = "regularization"
CONST_PARAM_NORM = "param_norm"
CONST_GRAD_NORM = "grad_norm"

CONST_DQN = "dqn"
CONST_PPO = "ppo"
CONST_VI = "valueiteration"

CONST_CE = "ce"
CONST_MSE = "mse"
CONST_MLE = "mle"
CONST_LIP = "lip"
CONST_CLIP = "clip"
CONST_SIGLIP = "siglip"
CONST_VIT = "vit"
CONST_BIAS_CORRECTION = "bias_correction"
COSNT_EFFICIENT_NET_B3 = "efficientnet_b3"
COSNT_EFFICIENT_NET_V2 = "efficientnet_v2"

CONST_TEXT_ENCODER = "text_encoder"
CONST_IMAGE_ENCODER = "image_encoder"
CONST_VISION_ENCODER = "vision_encoder"

CONST_PROPRIOCEPTION = "proprioception"
CONST_IMAGE = "image"
CONST_INPUTS = "inputs"
CONST_LABELS = "labels"

CONST_CONFUSION_MATRIX = "confusion_matrix"
CONST_SAME_PADDING = "SAME"
CONST_RNG = "rng"

VALID_ARCHITECTURE = [
    CONST_CNN,
    CONST_ENCODER_PREDICTOR,
    CONST_ENSEMBLE,
    CONST_ICL_GPT,
    CONST_MLP,
    CONST_RESNET,
    CONST_VIT,
    COSNT_EFFICIENT_NET_V2,
    COSNT_EFFICIENT_NET_B3,
]

VALID_TOKENIZER_TYPE = [
    CONST_CNN,
    CONST_MLP,
    CONST_RESNET,
]
VALID_POLICY_DISTRIBUTION = [
    CONST_BANG_BANG,
    CONST_DETERMINISTIC,
    CONST_GAUSSIAN,
    CONST_SOFTMAX,
    CONST_SQUASHED_GAUSSIAN,
]

VALID_EXPLORATION_POLICY = [
    CONST_SQUASHED_GAUSSIAN,
]

CONST_DISCRETE = "discrete"
CONST_CONTINUOUS = "continuous"
CONST_DEFAULT = "default"
CONST_POSE = "pose"

CONST_MANIPULATOR_LEARNING = "manipulator_learning"

CONST_TARGET_ENTROPY = "target_entropy"
