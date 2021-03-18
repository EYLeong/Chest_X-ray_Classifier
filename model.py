import torch
from torch import optim, nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import copy
import numpy as np
import utils

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
        
def train(model, device, loss_criterion, optimizer, train_loader, val_loader, epochs, patience, layer):
    best_loss = float("inf")
    early_stop = 0
    best_weights = None

    epoch_list = []
    training_loss = []
    validate_loss = []
    
    for i in range(epochs):
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

def load_model(model, optimizer, path, device):
    '''
    Saves model parameters instead of the whole model. 
    Requires first initialization of the model class to be fed as input to this function, in which
    state_dictionaries are loaded and returned
    '''

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optimizer.load_state_dict(checkpoint['opti_state_dict'])
    
    return model, optimizer

class Combined_Model:
    def __init__(self,fl_model,sl_model):
        self.fl_model = fl_model
        self.sl_model = sl_model

    def test_combine(self, device, test_loader, print_acc=False, return_results=False):

        self.fl_model.eval()
        self.sl_model.eval()

        correct = 0

        with torch.no_grad():
            for data, target in test_loader:

                data, target = data.to(device), target.to(device)
                
                fl_output = self.fl_model.forward(data)
                fl_pred = torch.round(fl_output)

                sl_output = self.sl_model.forward(data)
                sl_pred = torch.round(sl_output)

                # method to split the fl target and sl targets
                fl_target = target[:,:1]
                sl_target_covid = target[:,1:2] 
                sl_target_noncovid = target[:,2:3]  
                
                #First Layer Check 
                #for normal cases we manipulate so that 0 becomes 1, and 1 becomes 0           
                fl_pred_mod = copy.deepcopy(fl_pred)
                fl_pred_mod[fl_pred_mod==0] = 2
                fl_pred_mod[fl_pred_mod==1] = 0
                fl_pred_mod[fl_pred_mod==2] = 1

                equal_data_fl = torch.sum(torch.mul(fl_target.data,fl_pred_mod)).item()
                correct += equal_data_fl

                #Second Layer Check 
                #we copy another target tensor for covid cases and manipulate so that 0 becomes 1, and 1 becomes 0
                sl_pred_covid = copy.deepcopy(sl_pred)             
                sl_pred_covid[sl_pred_covid==0] = 2
                sl_pred_covid[sl_pred_covid==1] = 0
                sl_pred_covid[sl_pred_covid==2] = 1

                #Accounts for the second layer check only when first layer shows that it is a 1 = Infected
                sl_pred_covid = torch.mul(fl_pred,sl_pred_covid)
                sl_pred_non_covid = torch.mul(fl_pred,sl_pred)
                
                #Calculates the equal matches between the target and data
                equal_data_sl_covid = torch.sum(torch.mul(sl_target_covid.data,sl_pred_covid)).item()
                equal_data_sl_non_covid = torch.sum(torch.mul(sl_target_noncovid.data,sl_pred_non_covid)).item()
                     
                correct += equal_data_sl_covid
                correct += equal_data_sl_non_covid

                #Final Pred Results
                pred = torch.cat((fl_pred_mod,sl_pred_covid,sl_pred_non_covid),1) 

        if print_acc == True:
            print('Test set accuracy: ', 100. * correct / len(test_loader.dataset), '%')

        if return_results == True:
            return pred


def combine_models(fl_image_path, sl_image_path, device, L_RATE):
    
    fl_model = CNN().to(device)
    fl_optimizer = optim.Adam(fl_model.parameters(), lr=L_RATE)
    fl_model, _ = load_model(fl_model, fl_optimizer, fl_image_path, device)
    
    sl_model = CNN().to(device)
    sl_optimizer = optim.Adam(sl_model.parameters(), lr=L_RATE)
    sl_model, _ = load_model(sl_model, sl_optimizer, sl_image_path, device)

    return Combined_Model(fl_model, sl_model)