import torch
from torch import optim, nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import copy
import numpy as np
import utils

class CNN(nn.Module):
    def __init__(self, dropout=0.1):
        super(CNN,self).__init__()
        
        """
        Insert each layer blocks. Same architecture will be used for the first layer and second layer CNNs
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=dropout))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout(p=dropout))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout(p=dropout))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout(p=dropout))

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
        
def train(model, device, loss_criterion, optimizer, train_loader, val_loader, epochs, patience, layer):
    best_loss = float("inf")
    early_stop = 0
    best_weights = None

    epoch_list = []
    training_loss = []
    validate_loss = []
    
    for i in range(epochs):
        print('\n')
        print("Epoch {}".format(i+1))
        train_loss = train_epoch(model, device, loss_criterion, optimizer, train_loader)
        val_loss = validate(model, device, loss_criterion, val_loader)

        epoch_list.append(i+1)
        training_loss.append(train_loss)
        validate_loss.append(val_loss)
        
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

    #Generate the learning curves
    utils.generate_graph(epoch_list,training_loss,validate_loss, layer)

def train_epoch(model, device, loss_criterion, optimizer, train_loader):
    model.to(device)
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

    return (running_loss / counter)
        
def validate(model, device, loss_criterion, val_loader):
    model.to(device)
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
    
def test(model, device, test_loader):
    model.to(device)
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

def save_model(model, optimizer, path):
    checkpoint = {'state_dict': model.state_dict(),
                  'opti_state_dict': optimizer.state_dict()
                  }
    torch.save(checkpoint, path)

def load_model(lr, dropout, wd, path):
    '''
    Saves model parameters instead of the whole model. 
    Requires first initialization of the model class to be fed as input to this function, in which
    state_dictionaries are loaded and returned
    '''

    checkpoint = torch.load(path)
    model = CNN(dropout=dropout)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = optimizer.load_state_dict(checkpoint['opti_state_dict'])
    
    return model, optimizer

class Combined_Model:
    def __init__(self,fl_model,sl_model):
        self.fl_model = fl_model
        self.sl_model = sl_model
        
    def predict(self, device, data):
        self.fl_model.to(device)
        self.sl_model.to(device)
        self.fl_model.eval()
        self.sl_model.eval()
       
        batch_pred = torch.empty(0, 3).to(device)
        data = data.to(device)
        
        with torch.no_grad():
            for sample in data:
                sample = torch.unsqueeze(sample, 0)
                fl_output = self.fl_model.forward(sample)
                fl_pred = torch.round(fl_output)
                if fl_pred == 0:
                    batch_pred = torch.cat((batch_pred, torch.tensor([[1., 0., 0.]]).to(device)), 0)
                else:
                    sl_output = self.sl_model.forward(sample)
                    sl_pred = torch.round(sl_output)
                    if sl_pred == 0:
                        batch_pred = torch.cat((batch_pred, torch.tensor([[0., 1., 0.]]).to(device)), 0)
                    else:
                        batch_pred = torch.cat((batch_pred, torch.tensor([[0., 0., 1.]]).to(device)), 0)
        return batch_pred
    
    def predict_loader(self, device, loader):
        all_pred = torch.empty(0, 3).to(device)
        
        for data, labels in loader:
            all_pred = torch.cat((all_pred, self.predict(device, data)), 0)
        return all_pred
        

def load_combined(fl_model_path, sl_model_path, lr1, dropout1, wd1, lr2, dropout2, wd2):

    fl_model, _ = load_model(lr1, dropout1, wd1, fl_model_path)
    
    sl_model, _ = load_model(lr2, dropout2, wd2, sl_model_path)

    return Combined_Model(fl_model, sl_model)

def accuracy(predicted, actual):
    return ((predicted * actual).sum() / len(predicted)).item()

def precision(predicted, actual):
    classes = predicted.shape[1]
    true_pos = []
    all_pos = []
    for i in range(classes):
        true_pos.append(int((predicted[:, i] * actual[:, i]).sum().item()))
        all_pos.append(int(predicted[:,i].sum().item()))
    precision = [true_pos[i]/all_pos[i] for i in range(classes)]
    return precision
    
def recall(predicted, actual):
    classes = predicted.shape[1]
    true_pos = []
    relevant = []
    for i in range(classes):
        true_pos.append(int((predicted[:, i] * actual[:, i]).sum().item()))
        relevant.append(int(actual[:,i].sum().item()))
    recall = [true_pos[i]/relevant[i] for i in range(classes)]
    return recall