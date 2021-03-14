import torch
from torch import optim, nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import copy

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        """
        Insert each layer blocks. Same architecture will be used for the first layer and second layer CNNs
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.05))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.05))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.05))

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())

        self.fc = nn.Linear(512, 1)
        
        """
        Initialization of the Conv layers and FC layer using Kaiming initialization
        """
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.conv3.apply(init_weights)
        self.conv4.apply(init_weights)
        self.fc.apply(init_weights)
    
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        x = self.fc(x)

        return torch.sigmoid(x)
    
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
def train(model, device, loss_criterion, optimizer, train_loader, val_loader, epochs, patience):
    best_loss = float("inf")
    early_stop = 0
    best_weights = None
    
    for i in range(epochs):
        print("Epoch {}".format(i+1))
        train_epoch(model, device, loss_criterion, optimizer, train_loader)
        val_loss = validate(model, device, loss_criterion, val_loader)
        print()
        
        """
        Early Stopping 
        """
        if val_loss < best_loss:
            early_stop = 0
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
        else:
            early_stop += 1
                
        if early_stop == patience:
            model.load_state_dict(best_weights)
            break

def train_epoch(model, device, loss_criterion, optimizer, train_loader):
    model.train()
    
    running_loss = 0
    
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    counter = 0
    
    for batch_idx, (data, target) in enumerate(tk0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model.forward(data)
        
        target = target.argmax(dim=1, keepdim=True).float()
        
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        counter += 1
        tk0.set_postfix(loss=(running_loss / counter))
        
def validate(model, device, loss_criterion, val_loader):
    model.eval()

    correct = 0
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            target = target.argmax(dim=1, keepdim=True).float()
            
            output = model.forward(data)
            val_loss += loss_criterion(output, target).item()
            
            pred = torch.round(output)
            equal_data = torch.sum(target.data == pred).item()
            correct += equal_data

    print("Validation loss: {}".format(val_loss / len(val_loader)))
    print('Validation set accuracy: ', 100. * correct / len(val_loader.dataset), '%')
    
    return (val_loss / len(val_loader))
    
def test(model, device, test_loader, plot=False):
    model.eval()
    
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target = target.argmax(dim=1, keepdim=True).float()
            
            output = model.forward(data)
            pred = torch.round(output)

            equal_data = torch.sum(target.data == pred).item()
            correct += equal_data

    print('Test set accuracy: ', 100. * correct / len(test_loader.dataset), '%')