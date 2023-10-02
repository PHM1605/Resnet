import torchvision.transforms as transforms
from config import CLASSES, IMG_SIZE
from PIL import Image 
from utils import onnx_predict

if __name__ == "__main__":
    img = Image.open("./samples/full/4.AQUA/CT00004754CH_CCTB_MN23_Aqua202308151255521.jpg")
    img = transforms.Resize(IMG_SIZE)(img)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    output = onnx_predict(img_tensor, 'best.onnx')
    print(CLASSES[output[0]])