import torch
from utils import get_classes, get_num_classes

BATCH_SIZE = 8
CLASSES = get_classes()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
IMG_SIZE = (224, 224)
NUM_CLASSES = get_num_classes()
