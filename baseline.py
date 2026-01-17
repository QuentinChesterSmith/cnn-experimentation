import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutional_layers = nn.Sequential(
            # 1 Feature Map 28x28
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, padding="same"),
            nn.MaxPool2d(2),
            # 5 Feature Maps 14x14
            nn.Conv2d(in_channels=5, out_channels=15, kernel_size=5, padding="same"),
            nn.MaxPool2d(2),
            # 15 Feature Maps 7x7
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(735, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        
    def forward(self, x):
        x = self.convolutional_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fully_connected_layers(x)
        return x
    
    def pred(self, x):
        return self.forward(x)

def calc_acc(dl, model, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (X, y) in dl:
            X, y = X.to(device), y.to(device)
            pred = model.pred(X)
            outputs = torch.argmax(pred, dim=1)
            correct += (outputs == y).sum().item()
            total += y.size(0)
        acc = correct/total
    return acc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)

lengths = [int(len(dataset)*.8), int(len(dataset)*.1), int(len(dataset)*.1)]
trainset, testset, valset = random_split(dataset, lengths)
# 80% Train, 10% Validation, 10% Test

train_dl = DataLoader(trainset, batch_size=64, shuffle=True)
val_dl = DataLoader(valset, batch_size=64)

model = CNN().to(device)

lr = 0.005
epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

print("Starting Training")
for epoch in range(epochs):
    running_loss = 0
    batches = 0
    train_acc = None
    val_acc = None

    model.train()
    for (X, y) in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        batches+=1
        running_loss+=loss.item()
    
    # Evaluation Metrics
    train_acc = calc_acc(train_dl, model, device)
    val_acc = calc_acc(val_dl, model, device)
        
    print(f"Epoch {epoch+1}: Loss {running_loss/batches}")
    print(f"Train Accuracy: {train_acc:.4f} Validation Accuracy: {val_acc:.4f}")
