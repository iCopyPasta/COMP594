
# coding: utf-8

# ## Imports

# In[1]:


#from __future__ import print_function, division

import torch
import torch.nn.parallel
import torch.utils
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
from PIL import Image
import DrawingWithTensors

from torchvision.transforms import ToPILImage
#from IPython.display import Image
#to_img = ToPILImage()
#from IPython.display import Image

#plt.ion()   # interactive mode

#original code for training: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

#imports related to fully convolutional network
import torchfcn

#original paths for FCNs:
#/home/peo5032/data/models/chainer/fcn16s_from_caffe.npz
# calling torchfcn.models.FCN16s.pretrained_model yields:
# might need to call download on it first: torchfcn.models.FCN16s.download()
#'/home/peo5032/data/models/pytorch/fcn16s_from_caffe.pth'


# In[2]:


import os
import argparse

# initiate the parser
parser = argparse.ArgumentParser(description = "List of options to run application when creating custom datset")

parser = argparse.ArgumentParser()  
parser.add_argument("-V", "--version", help="show program version", action="store_true")
parser.add_argument("-b", "--batch", help="batch size in each epoch")
parser.add_argument("-e", "--epoch", help="number of epochs for training")
parser.add_argument("-r", "--root_folder", help="destination for root folder")
parser.add_argument("-i", "--iteration", help="which generation number we are using")
parser.add_argument("-t", "--training", help="True/False to load FCN weights on start")
parser.add_argument("-w", "--weights", help="full path to save weights")
parser.add_argument("-k", "--resume", help="full path to resume training use weights")


# In[3]:


PRETRAINED_PATH = '/home/peo5032/data/models/pytorch/fcn16s_from_caffe.pth'
SAVE_LOCATION = "/home/peo5032/Documents/COMP594/model.pt"
LOAD_LOCATION = "/home/peo5032/Documents/COMP594/model.pt"

NUM_CLASSES = 7
EPOCHS = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu" #just for testing for sunlab
imageSize = 400
batchSize = 1
iteration = "1"
newTraining = False

#change values if user specifies non-default values
args = parser.parse_args()

# check for --version or -V
if args.version:  
    print("this is version 0.1")
    
if args.batch: 
    print("batch size is set to", args.batch)
    batchSize = int(args.batch)

if args.epoch: 
    print("number of epochs is set to", args.epoch)
    EPOCHS = int(args.epoch)
    
if args.root_folder:  
    if os.path.exists(root_folder):
        ROOT = root_folder
    print("destination was", args.root_folder)
    
if args.iteration:
    print("iteration is set to", args.iteration)
    iteration = args.iteration
    
if args.training:
    if args.training.lower() == "true":
        print("training is set to true")
        newTraining = True
        
if args.weights:
    print("save location is set to", args.weights)
    SAVE_LOCATION = args.weights
    
if args.resume:
    print("load location is set to", args.resume)
    LOAD_LOCATION = args.resume

#TODO in arguments
# root folder location
# saved weights location


# In[4]:


from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# In[5]:


#https://github.com/GautamSridhar/FCN-implementation-on-Pytorch/blob/master/DiceLoss.py
# deleted

#https://github.com/iCopyPasta/Pytorch-UNet/blob/master/dice_loss.py
from torch.autograd import Function, Variable

class DiceCoeff(torch.nn.Module):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_tensors
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter)                          / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


    def dice_coeff(input, target):
        """Dice coeff for batches"""
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            s = s + DiceCoeff().forward(c[0], c[1])

        return s / (i + 1)


# In[6]:


import torch.nn as nn
import torch.nn.functional as F


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# ## Load Data

# In[7]:


data_transforms = transforms.Compose([transforms.Resize([imageSize,imageSize]),
                                      transforms.ToTensor()
                                     ])

# instantiate the dataset and dataloader
data_dir = '/home/peo5032/Documents/COMP594/input/gen'+iteration
dataset = ImageFolderWithPaths(data_dir, transform=data_transforms) # our custom dataset
dataloaders = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle=True, num_workers=1)


# iterate over data
#for inputs, labels, paths in dataloader:
#    # use the above variables freely
#    print(inputs, labels, paths)

#groundTruth = tensor
#label = tensor[0,0]
#path = tuple list, access each via path[index]

new_road_factory = DrawingWithTensors.datasetFactory()


# ## Training Routine without Validation Steps

# In[8]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=4):
    since = time.time()

    best_model_wts = model.state_dict().copy()
    best_acc = 0.0
    
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        
        running_loss = 0.0
        running_corrects = 0
               
        #BATCH TUPLE
        inputs, labels, paths = next(iter(dataloaders))
        inputs.to(device)
                
        #build ground-truth batch tensor
        for locations in paths:
            i = 0
            #dtype=torch.int64
            labels = torch.zeros(batchSize,NUM_CLASSES,imageSize,imageSize, dtype = torch.float32)
            labels[i] = torch.load(locations.replace(".png", ".pt").replace("roads", "tensor_values")) #manually fetch your own tensor values here somehow? 
            i += 1
            
        

        # forward
        # track history if only in train
        # TODO: ENSURE OUTPUTS AND GROUNDTRUTH ARE THE SAME
        with torch.set_grad_enabled(True):
            #build input-truth batch tensor
            outputs = model(inputs)
            loss = criterion(outputs, labels) #ground truth comparison

            # backward + optimize 
            loss.backward()
            optimizer.step()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            
        # statistics
        #epoch_loss = loss.item() * inputs.size(0) # unsure what this part is
        epoch_loss = loss.item() # unsure what this part is
        print('epoch loss:',epoch_loss)
        
        
        #running_corrects += torch.sum(preds == labels.data) # unsure what this part is

        #epoch_loss = running_loss / dataset_sizes[phase]
        #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
        #epoch_loss = running_loss / dataset_sizes[phase]
        #epoch_acc = running_corrects.double() / dataset_sizes[phase]

        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #phase, running_loss, ))

        # deep copy the model
        #if phase == 'val' and epoch_acc > best_acc:
        #    best_acc = epoch_acc
        #    best_model_wts = copy.deepcopy(model.state_dict())
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    torch.save(model, SAVE_LOCATION)
    
    return model


# In[9]:


def showInferenceOnImage(img, tensor, class_label, threshold, classMap):
    IMAGE_SIZE = 400
    imgTMP = img.copy()
    imgMap = imgTMP.load()
    class_type_corresponding_channel = classMap[class_label]
    print("index for channel", class_label, ":", class_type_corresponding_channel)    
    for i in range(0, IMAGE_SIZE):
        for j in range(0, IMAGE_SIZE):
            if tensor[class_type_corresponding_channel, i,j] >= threshold:
                #show class label in white
                imgMap[i,j] = (0,0,0)
        
    return imgTMP


# ## Load Pretrained Model Weights

# In[10]:


if newTraining is True:
    model = torchfcn.models.FCN16s()
    model.load_state_dict(torch.load(PRETRAINED_PATH))
    
else:
    model = torch.load(LOAD_LOCATION)
    
model = model.to(device)


# ## Change Architecture for New Classes and New Training

# In[11]:


if newTraining is True:  
    model.score_fr = torch.nn.Conv2d(4096, NUM_CLASSES , kernel_size=(1, 1),
                                     stride=(1, 1))
    torch.nn.init.uniform_(model.score_fr.weight, a=0, b=0.05)
    torch.nn.init.uniform_(model.score_fr.bias, a=0, b=0.05)
    #model.score_fr.weight.data.fill_(0.10)
    #model.score_fr.bias.data.fill_(0.00)

    model.score_pool4 = torch.nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1),
                                        stride=(1, 1))
    torch.nn.init.uniform_(model.score_pool4.weight, a=0, b=0.05)
    torch.nn.init.uniform_(model.score_pool4.bias, a=0, b=0.05)
    #model.score_pool4.weight.data.fill_(0.10)
    #model.score_pool4.bias.data.fill_(0.00)

    model.upscore2 = torch.nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, kernel_size=(4, 4),
                                              stride=(2, 2),bias=False)
    torch.nn.init.uniform_(model.upscore2.weight, a=0, b=0.05)
    #model.upscore2.weight.data.fill_(0.10)

    model.upscore16 = torch.nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, kernel_size=(32, 32),
                                               stride=(16, 16),bias=False)
    torch.nn.init.uniform_(model.upscore16.weight, a=0, b=0.05)
    #model.upscore16.weight.data.fill_(0.10)
    
    torch.save(model, SAVE_LOCATION)


# ## Training and Results

# In[12]:


#criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()
criterion = torch.nn.BCEWithLogitsLoss()
#criterion = torch.nn.NLLLoss()
#criterion = DiceCoeff()
#criterion = SoftDiceLoss()

# Observe default choices, except using amsgrad version of Adam
optimizer_ft = optim.Adam(model.parameters(),amsgrad=True)

# Osscilate between high and low learning rates
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, 7)


# In[14]:


model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS)

