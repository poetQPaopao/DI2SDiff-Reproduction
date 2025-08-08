"""
MotionSense Dataset Configuration
"""

class MotionSenseConfig:
    """MotionSense数据集和模型配置"""
    
    # 数据配置
    DATA_DIR = '/root/.jupyter/lab/workspaces/motion-sense/data/'
    DATASET_NAME = 'motionsense'
    
    # 数据参数
    NUM_FEATURES = 12  # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    NUM_CLASSES = 6    # dws, ups, wlk, jog, sit, std
    NUM_SUBJECTS = 24
    WINDOW_SIZE = 128
    OVERLAP = 0.5
    SAMPLE_RATE = 50  # Hz
    
    # 活动标签
    ACTIVITY_NAMES = ["dws", "ups", "wlk", "jog", "sit", "std"]
    ACTIVITY_LABELS = {
        "dws": 0,    # downstairs
        "ups": 1,    # upstairs  
        "wlk": 2,    # walking
        "jog": 3,    # jogging
        "sit": 4,    # sitting
        "std": 5     # standing
    }
    
    # 试验编码
    TRIAL_CODES = {
        "dws": [1, 2, 11], 
        "ups": [3, 4, 12], 
        "wlk": [7, 8, 15],
        "jog": [9, 16], 
        "sit": [5, 13], 
        "std": [6, 14]
    }
    
    # TS-TCC模型配置
    class TSTCCConfig:
        # 网络架构
        INPUT_CHANNELS = 12  # NUM_FEATURES
        MID_CHANNELS = 64
        FINAL_OUT_CHANNELS = 128
        NUM_CLASSES = 6  # NUM_CLASSES
        DROPOUT = 0.35
        
        # 对比学习
        TEMPERATURE = 0.2
        PROJECTION_DIM = 50
        
        # 训练参数
        BATCH_SIZE = 64
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 3e-4
        NUM_EPOCHS = 100
        PATIENCE = 10
        
        # 优化器
        BETA1 = 0.9
        BETA2 = 0.99
        
        # 数据增强
        AUGMENTATIONS = ['jitter', 'scaling', 'rotation', 'permutation']
        
        # 调度器
        USE_SCHEDULER = True
        SCHEDULER_PATIENCE = 5
        SCHEDULER_FACTOR = 0.5
    
    # Diffusion模型配置
    class DiffusionConfig:
        # 模型架构
        INPUT_CHANNELS = 12  # NUM_FEATURES
        DIM = 64
        DIM_MULTS = (1, 2, 4, 8)
        COND_DIM = 128
        
        # 扩散参数
        TIMESTEPS = 1000
        BETA_SCHEDULE = 'cosine'
        OBJECTIVE = 'pred_noise'
        
        # 训练参数
        BATCH_SIZE = 32
        LEARNING_RATE = 1e-4
        NUM_EPOCHS = 200
        SAVE_EVERY = 10
        
        # 生成参数
        SAMPLE_BATCH_SIZE = 16
        NUM_SAMPLES = 1000
    
    # 数据预处理
    class PreprocessConfig:
        NORMALIZE = True
        NORMALIZATION_METHOD = 'standard'  # 'standard' or 'minmax'
        FILTER_ACTIVITIES = None  # None表示使用所有活动
        MIN_SEQUENCE_LENGTH = 128  # WINDOW_SIZE
        
        # 数据分割
        TEST_SIZE = 0.2
        VAL_SIZE = 0.1
        SPLIT_METHOD = 'trial'  # 'trial' or 'random' or 'subject'
        RANDOM_STATE = 42
    
    # 评估配置
    class EvalConfig:
        METRICS = ['accuracy', 'f1', 'precision', 'recall']
        AVERAGE = 'macro'
        
        # 可视化
        PLOT_CONFUSION_MATRIX = True
        PLOT_TSNE = True
        PLOT_TRAINING_CURVES = True
        
        # 生成样本评估
        EVAL_GENERATED_SAMPLES = True
        PEARSON_CORRELATION = True
        DTW_DISTANCE = True
    
    # 文件路径
    class PathConfig:
        DATA_RAW = '/root/.jupyter/lab/workspaces/motion-sense/data/'  # DATA_DIR
        DATA_PROCESSED = './data/processed/motionsense/'
        MODELS = './models/motionsense/'
        LOGS = './logs/motionsense/'
        RESULTS = './results/motionsense/'
        CHECKPOINTS = './checkpoints/motionsense/'
        
        # 保存的数据文件
        TRAIN_DATA = 'X_train.npy'
        TEST_DATA = 'X_test.npy'
        VAL_DATA = 'X_val.npy'
        TRAIN_LABELS = 'y_train.npy'
        TEST_LABELS = 'y_test.npy'
        VAL_LABELS = 'y_val.npy'
        DATASET_INFO = 'dataset_info.npy'

# 为方便使用创建全局配置实例
config = MotionSenseConfig()
tstcc_config = MotionSenseConfig.TSTCCConfig()
diffusion_config = MotionSenseConfig.DiffusionConfig()
preprocess_config = MotionSenseConfig.PreprocessConfig()
eval_config = MotionSenseConfig.EvalConfig()
path_config = MotionSenseConfig.PathConfig()
