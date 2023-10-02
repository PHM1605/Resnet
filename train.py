import torch
import torch.nn as nn
import torch.optim as optim
from config import BATCH_SIZE, DEVICE, EPOCHS, NUM_CLASSES
from dataset import PosmDataset
from model import ResNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import onnx_convert

def train_fn(train_loader, model, optimizer, loss_fn):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for batch_idx, (X, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        X = X.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, target)
        train_running_loss += loss.item()
        train_running_correct += (torch.argmax(preds, dim=1)==torch.argmax(target, dim=1)).sum().item()
        loss.backward()
        optimizer.step()
        
    epoch_loss = train_running_loss / (batch_idx+1)
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc

def val_fn(val_loader, model):
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for (X, y) in val_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            preds = model(X)
            val_correct += (torch.argmax(preds, dim=1)==torch.argmax(y, dim=1)).sum().item()
    val_acc = 100. * (val_correct / len(val_loader.dataset))
    return val_acc

if __name__ == "__main__":
    model = ResNet(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    train_dataset = PosmDataset(img_dir="samples/train_small/*/*.jpg")
    val_dataset = PosmDataset(img_dir="samples/val/*/*.jpg")
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    best_acc = 0
    for epoch in range(EPOCHS):
        epoch_loss, epoch_acc = train_fn(train_loader, model, optimizer, loss_fn)
        print(f"Training loss: {epoch_loss:.3f}, training acc: {epoch_acc:.3f}")
        val_acc = val_fn(val_loader, model)
        print(f"Validation acc: {val_acc:.3f}")
        if val_acc > best_acc:
            onnx_convert(model, dummy_input=val_dataset[0][0].unsqueeze(0).to(DEVICE), model_name='best.onnx')
            best_acc = val_acc
            print(f"New best.onnx model found with {best_acc:.3f} accuracy")
        if epoch_acc > 99.0 and val_acc > 99.0:
            break
        print('-'*50)
