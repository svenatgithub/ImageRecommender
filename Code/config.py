import os

# Base paths
BASE_PATH = r"D:\raw_data\Code"
PATH_TO_SSD = r"D:\raw_data\data"


# Database path
DB_PATH = os.path.join(BASE_PATH, 'collected_data', 'metadata.db')

# Data paths
COLLECTED_DATA_PATH = os.path.join(BASE_PATH, 'collected_data')

# Separate paths for RGB and HSV CSV files
RGB_PATH = os.path.join(COLLECTED_DATA_PATH, 'rgb_histograms.csv')
HSV_PATH = os.path.join(COLLECTED_DATA_PATH, 'hsv_histograms.csv')

PICKLE_PATH = os.path.join(BASE_PATH, 'pickles', 'checkpoint.pkl')

# Ensure the pickle directory exists
os.makedirs(PICKLE_PATH, exist_ok=True)

# Processing parameters
COMPRESS_QUALITY = 75
CHUNK_SIZE = 100
TOTAL_IMAGES = 444.891 # Total number of images to process

# Image processing parameters
MAX_IMAGE_SIZE = (512, 512)  
TARGET_IMAGE_SIZE = (224, 224)  # Resizing the images For EfficientNet b0

ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.svg')

# Checkpoint parameters
CHECKPOINT_INTERVAL = 1000  # Interval at which checkpoints are saved
CHECKPOINT_PATH = os.path.join(PICKLE_PATH, 'checkpoint.pkl')  # Path to save checkpoint data
