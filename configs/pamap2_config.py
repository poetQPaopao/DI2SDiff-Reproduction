"""
PAMAP2 Dataset Configuration
"""

class PAMAP2Config:
    """PAMAP2数据集和模型配置"""
    
    # 数据配置
    DATA_DIR = '/root/.jupyter/lab/workspaces/PAMAP2_Dataset/Protocol'
    DATASET_NAME = 'pamap2'
    
    # 数据参数
    NUM_FEATURES = 16  # 多传感器IMU数据 + 心率
    NUM_CLASSES = 8    # lying, sitting, standing, walking, running, cycling, nordic_walking, stairs
    NUM_SUBJECTS = 9   # subject01-subject09
    WINDOW_SIZE = 128
    OVERLAP = 0.5
    SAMPLE_RATE = 100  # Hz
    
    # 活动标签映射
    ACTIVITY_NAMES = [
        'lying', 'sitting', 'standing', 'walking', 
        'running', 'cycling', 'nordic_walking', 'stairs'
    ]
    
    ACTIVITY_LABELS = {
        'lying': 0,
        'sitting': 1, 
        'standing': 2,
        'walking': 3,
        'running': 4,
        'cycling': 5,
        'nordic_walking': 6,
        'stairs': 7
    }
    
    # 原始PAMAP2活动ID到我们的标签的映射
    PAMAP2_ACTIVITY_MAP = {
        1: 0,  # lying
        2: 1,  # sitting
        3: 2,  # standing
        4: 3,  # walking
        5: 4,  # running
        6: 5,  # cycling
        7: 6,  # Nordic walking
        12: 7, # ascending stairs
        13: 7, # descending stairs (合并为stairs)
        16: 3, # vacuum cleaning (归类为walking)
        17: 4, # ironing (归类为running)
    }
    
    # 选择的传感器列 (总共16个特征)
    SELECTED_COLUMNS = [
        # Chest IMU - 加速度计 (3轴) - 列17,18,19
        17, 18, 19,
        # Chest IMU - 陀螺仪 (3轴) - 列20,21,22
        20, 21, 22,
        # Chest IMU - 磁力计 (3轴) - 列23,24,25
        23, 24, 25,
        # Hand IMU - 加速度计 (3轴) - 列4,5,6
        4, 5, 6,
        # Ankle IMU - 加速度计 (3轴) - 列30,31,32
        30, 31, 32,
        # 心率 (列2)
        2,
    ]
    
    # TS-TCC模型配置
    class TSTCCConfig:
        # 网络架构
        INPUT_CHANNELS = 16  # NUM_FEATURES
        MID_CHANNELS = 64
        FINAL_OUT_CHANNELS = 128
        NUM_CLASSES = 8  # NUM_CLASSES
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
        INPUT_CHANNELS = 16  # NUM_FEATURES
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
        MIN_LABEL_CONSISTENCY = 0.8  # 窗口内标签一致性要求
        
        # 数据分割
        TEST_SIZE = 0.2
        VAL_SIZE = 0.1
        SPLIT_METHOD = 'random'  # 'random' or 'subject'
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
        DATA_RAW = '/root/.jupyter/lab/workspaces/PAMAP2_Dataset/Protocol'  # DATA_DIR
        DATA_PROCESSED = './data/processed/pamap2/'
        MODELS = './models/pamap2/'
        LOGS = './logs/pamap2/'
        RESULTS = './results/pamap2/'
        CHECKPOINTS = './checkpoints/pamap2/'
        
        # 保存的数据文件
        TRAIN_DATA = 'X_train.npy'
        TEST_DATA = 'X_test.npy'
        VAL_DATA = 'X_val.npy'
        TRAIN_LABELS = 'y_train.npy'
        TEST_LABELS = 'y_test.npy'
        VAL_LABELS = 'y_val.npy'
        SUBJECTS_TRAIN = 'subjects_train.npy'
        SUBJECTS_TEST = 'subjects_test.npy'
        DATASET_INFO = 'dataset_info.npy'

# 为方便使用创建全局配置实例
config = PAMAP2Config()
tstcc_config = PAMAP2Config.TSTCCConfig()
diffusion_config = PAMAP2Config.DiffusionConfig()
preprocess_config = PAMAP2Config.PreprocessConfig()
eval_config = PAMAP2Config.EvalConfig()
path_config = PAMAP2Config.PathConfig()
