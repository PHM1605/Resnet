import numpy as np
import onnxruntime, torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms 
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
def get_classes():
    with open("classes_posm.txt", "r") as f:
        lines = f.readlines()
    classes = [l.rstrip("\n") for l in lines]
    return classes

def get_num_classes():
    return len(get_classes())

def onnx_convert(model, dummy_input, model_name):
    model.eval()
    torch.onnx.export(model, dummy_input, model_name, verbose=False, input_names=["input"], output_names=["output"], export_params=True)
    return model_name

def onnx_predict(img, model_name):
    onnx_session = onnxruntime.InferenceSession(model_name, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(img)}
    onnx_output = onnx_session.run(None, onnx_inputs)
    img_label = np.argmax(onnx_output[0], axis=1)
    return img_label

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()  
    else:
        return tensor.cpu().numpy()