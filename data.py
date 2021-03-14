import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os

class Lung_Dataset(Dataset):
    """
    Lung Dataset Consisting of Infected and Non-Infected.
    """

    def __init__(self, purpose, verbose=0):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        
        Parameter:
        -purpose variable should be set to a string of either 'train', 'test' or 'val'
        -verbose takes an int of either 0,1 or 2. 0 will only differentiate between normal and infected, 1 will differentiate
            between normal, covid and non-covid while 2 will only differentiate between covid and non-covid
        """
        self.purpose = purpose
        self.verbose = verbose
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
            
        # The dataset has been split in training, testing and validation datasets
        self.groups = ['train', 'test', 'val']
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': './dataset/train/normal/',
                              'train_infected': './dataset/train/infected/',
                              'train_infected_covid': './dataset/train/infected/covid',
                              'train_infected_non_covid': './dataset/train/infected/non-covid',
                              'test_normal': './dataset/test/normal/',
                              'test_infected': './dataset/test/infected/',
                              'test_infected_covid': './dataset/test/infected/covid',
                              'test_infected_non_covid': './dataset/test/infected/non-covid',
                              'val_normal': './dataset/val/normal/',
                              'val_infected': './dataset/val/infected/',
                              'val_infected_covid': './dataset/val/infected/covid',
                              'val_infected_non_covid': './dataset/val/infected/non-covid'}
        
        self.dataset_numbers = {}
        
        # Consider normal and infected only
        if verbose == 0:
            self.classes = {0: 'normal', 1: 'infected'}
            
            #Populate self.dataset_numbers
            for condition in self.classes.values():
                key = "{}_{}".format(self.purpose, condition)
                if condition == "normal":
                    file_path = self.dataset_paths[key]
                    count = len(os.listdir(file_path))
                    self.dataset_numbers[key] = count
                else:
                    key1 = key + "_covid"
                    key2 = key + "_non_covid"
                    file_path1 = self.dataset_paths[key1]
                    file_path2 = self.dataset_paths[key2]
                    count1 = len(os.listdir(file_path1))
                    count2 = len(os.listdir(file_path2))
                    count = count1 + count2
                    self.dataset_numbers[key] = count
                       
        #Consider normal, covid and non-covid
        elif verbose == 1:
            self.classes = {0: 'normal', 1: 'covid', 2: 'non_covid'}
        
            #Populate self.dataset_numbers
            for condition in self.classes.values():
                if condition == "normal":
                    key = "{}_{}".format(self.purpose, condition)
                    file_path = self.dataset_paths[key]
                    count = len(os.listdir(file_path))
                    self.dataset_numbers[key] = count
                else:
                    key = "{}_infected".format(self.purpose)
                    key1 = key + "_covid"
                    key2 = key + "_non_covid"
                    file_path1 = self.dataset_paths[key1]
                    file_path2 = self.dataset_paths[key2]
                    count1 = len(os.listdir(file_path1))
                    count2 = len(os.listdir(file_path2))
                    self.dataset_numbers[key1] = count1
                    self.dataset_numbers[key2] = count2
                
        #Consider covid and non-covid
        elif verbose == 2:
            self.classes = {0: 'covid', 1 :'non_covid' }

            #Populate self.dataset_numbers
            for condition in self.classes.values():
                key = "{}_infected".format(self.purpose)
                key1 = key + "_covid"
                key2 = key + "_non_covid"
                file_path1 = self.dataset_paths[key1]
                file_path2 = self.dataset_paths[key2]
                count1 = len(os.listdir(file_path1))
                count2 = len(os.listdir(file_path2))
                self.dataset_numbers[key1] = count1
                self.dataset_numbers[key2] = count2
            
        else:
            err_msg  = "Verbose argument only takes in an int of either 0,1 or 2"
            raise TypeError(err_msg)
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the Lung {} Dataset in the 50.039 Deep Learning class project".format(self.purpose)
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_numbers.items():
            if key != 'infected':
                file_path = self.dataset_paths[key]
            else:
                file_path = self.dataset_paths
            msg += " - {}, in folder {}: {} images.\n".format(key, file_path, val)
        print(msg)
        
        
    def open_img(self, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        group_val = self.purpose
        err_msg = "Error - class_val variable should be set to 'normal', 'infected', 'covid' or 'non_covid'."
        assert class_val in self.classes.values(), err_msg
        
        if class_val == 'covid' or class_val == 'non_covid':
            class_val = 'infected_' + class_val
            
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        if class_val != "infected":
            path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        else:
            covid_count = len(os.listdir(self.dataset_paths['{}_{}_covid'.format(group_val, class_val)]))
            if index_val < covid_count:
                path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}_covid'.format(group_val, class_val)], index_val)
            else:
                index_val = index_val - covid_count
                path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}_non_covid'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im
    
    
    def show_img(self, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        # Open image
        im = self.open_img(class_val, index_val)
        
        # Display
        plt.imshow(im)
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        #If we only have 2 classes
        if self.verbose == 0 or self.verbose == 2:
            first_val = int(list(self.dataset_numbers.values())[0])
            if index < first_val:
                class_val = self.classes[0]
                label = torch.Tensor([1, 0])
            else:
                class_val = self.classes[1]
                index = index - first_val
                label = torch.Tensor([0, 1])
            im = self.open_img(class_val, index)
            im = transforms.functional.to_tensor(np.array(im)).float()
          
        #If we have 3 classes to consider
        elif self.verbose == 1:
            first_val = int(list(self.dataset_numbers.values())[0])
            second_val = int(list(self.dataset_numbers.values())[1])
            if index < first_val:
                class_val = self.classes[0]
                label = torch.Tensor([1, 0, 0])
            elif index >= first_val and index < first_val + second_val:
                index = index - first_val
                class_val = self.classes[1]
                label = torch.Tensor([0,1,0])
            else:
                index = index-(first_val + second_val)
                class_val = self.classes[2]
                label = torch.Tensor([0,0,1])
            im = self.open_img(class_val, index)
            im = transforms.functional.to_tensor(np.array(im)).float()
                
        else:
            raise TypeError("Verbose value is not 0,1 or 2")
        return im, label