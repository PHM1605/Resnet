import glob, os, torch
import torch.nn as nn
import torchvision.transforms as transforms
from config import CLASSES, IMG_SIZE, NUM_CLASSES
from PIL import Image
from torch.utils.data import Dataset
from utils import Compose

class PosmDataset(Dataset):
    def __init__(self, img_dir):
        self.img_path = glob.glob(img_dir)
        self.transforms = Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = Image.open(img_path)
        folder_name = os.path.dirname(img_path).split('/')[-1]
        label = torch.nn.functional.one_hot( torch.tensor(CLASSES.index(folder_name)), num_classes=NUM_CLASSES )

        return self.transforms(img), label.float()

if __name__ == "__main__":
    img_dir = os.path.join("samples/VSC1000L/train", "*/*.jpg")
    dataset = LeftRightDataset(img_dir)
    X, y = dataset[0]
    print(X.shape, y)