
# coding: utf-8

# ## Imports

# In[1]:


#from __future__ import print_function, division

import torch
import torch.nn.parallel
import torch.utils
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
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
import math
import os
import sys

from torchvision.transforms import ToPILImage
#from IPython.display import Image
#to_img = ToPILImage()
#from IPython.display import Image

#plt.ion()   # interactive mode

#original code for training: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

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
parser.add_argument("-t", "--training", help="full path to load FCN weights on start")
parser.add_argument("-w", "--weights", help="full path to save weights")
parser.add_argument("-c", "--pickup", help="full path to resume training use weights")
parser.add_argument("-p", "--picture", help="picture dimensions")


# In[3]:


PRETRAINED_PATH = '/home/peo5032/data/models/pytorch/fcn16s_from_caffe.pth'
SAVE_LOCATION = "/home/peo5032/Documents/COMP594/model2.pt"
LOAD_LOCATION = "/home/peo5032/Documents/COMP594/model2.pt"

NUM_CLASSES = 7
EPOCHS = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu" #just for testing for sunlab
imageSize = 416
batchSize = 1
iteration = "1"
newTraining = False

#change values if user specifies non-default values
#'-t','True','-i','2','-e','325','-b','1', '-w', '/home/peo5032/Documents/COMP594/model2.pt', '-p','416'
args = parser.parse_args([])

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
    

if args.weights:
    print("save location is set to", args.weights)
    os.makedirs(args.weights, exist_ok=True)
    SAVE_LOCATION = args.weights
    
    

if args.pickup:
    print("load location is set to", args.pickup)
    LOAD_LOCATION = args.pickup
    
    
if args.training:
    if args.training.lower() == "true":
        print("training is set to true")
        newTraining = True
        
        
if args.picture:
    print("picture size to train on is", args.picture)
    imageSize = int(args.picture)
        

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

#https://github.com/milesial/Pytorch-UNet
import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
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

        input, target = self.saved_variables
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


#https://pythonexample.com/code/dice-loss-pytorch/
    
class DICELossMultiClass(torch.nn.Module):
 
    def __init__(self):
        super(DICELossMultiClass, self).__init__()
 
    def forward4(self, pred, targs):
        pred = (pred>0).float()
        return 2. * (pred*targs).sum() / (pred+targs).sum()
    
    #https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183#file-dice_coeff_loss-py-L3
    def forward3(self, pred, target):
        """This definition generalize to real valued pred and target vector.
    This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """

        smooth = .0000001

        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    
    
    #https://github.com/pytorch/pytorch/issues/1249
    def forward2(self,input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return ((2. * intersection + smooth) /  (iflat.sum() + tflat.sum() + smooth))
    
    def forward(self, input, target):
        #self.save_for_backward(input, target)
        eps = 0.0001
        #self.inter = torch.dot(input.view(-1), target.view(-1))
        self.inter = 2. * torch.dot(input.abs().view(-1), target.view(-1))
        self.union = torch.sum(torch.mul(input,input)) + torch.sum(target) 

        t = self.inter / self.union.float()
        return 1-t


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
    best_model = None
    best_loss = math.inf

    for epoch in range(1,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        model.train()  # Set model to training mode

        epoch_loss = 0
   
        #BATCH TUPLE
        inputs, labels, paths = next(iter(dataloaders))
        inputs.to(device)
        #print(inputs.size())
                
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

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # backward + optimize 
            loss.backward()
            optimizer.step()
            
            # statistics
            epoch_loss += loss.item()
        
        print('Epoch finished- dice: {}'.format(1-epoch_loss))
        print('Epoch finished-loss: {}'.format(epoch_loss))

        #save best copy of  model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model, SAVE_LOCATION.replace("model", "model_best"))
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed //3600,
                                                                time_elapsed // 60, time_elapsed % 60), flush=True)


    return model


# In[9]:


def showInferenceOnImage(img, tensor, class_label, threshold, classMap):
    IMAGE_SIZE = 416
    imgTMP = img.copy()
    imgMap = imgTMP.load()
    class_type_corresponding_channel = classMap[class_label]
    print("index for channel", class_label, ":", class_type_corresponding_channel)    
    for i in range(0, IMAGE_SIZE):
        for j in range(0, IMAGE_SIZE):
            if tensor[class_type_corresponding_channel, i,j] > threshold:
                #show class label in white
                imgMap[i,j] = (0,0,0)
        
    return imgTMP


# ## Load Pretrained Model Weights

# In[10]:


#imports related to UNet
from unet_models import *

if newTraining:
    model = UNet16(num_classes=7, num_filters=32, pretrained=True, is_deconv=True)
    
else:
    model = torch.load(LOAD_LOCATION)
    
model = model.to(device)


# In[11]:


item1 = torch.ones(1,2,2,2)
item2 = torch.ones(1,2,2,2)

crit = DICELossMultiClass()
#crit = torch.nn.BCELoss()
loss = crit(item1, item2)
print(loss.item())


# ## Training and Results

# In[12]:


#criterion = torch.nn.BCEWithLogitsLoss()
#criterion = torch.nn.BCELoss()
#criterion = DiceCoeff()
criterion = DICELossMultiClass()

# Observe default choices, except using amsgrad version of Adam
optimizer_ft = optim.Adam(model.parameters(),amsgrad=True)

# Osscilate between high and low learning rates
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, 7)

try:
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS)

except KeyboardInterrupt:
    torch.save(model, SAVE_LOCATION.replace("model",'INTERRUPTED'))
    print('Saved interrupt', flush=True)


# In[13]:


#Show results in meanwhile
#img = Image.open("/home/peo5032/Documents/COMP594/input/gen2/roads/100.png")
#test_tensor = torch.load("/home/peo5032/Documents/COMP594/input/gen2/tensor_values/100.pt")
#inputs = torch.zeros(1,3, imageSize, imageSize)


#inputs[0] = transforms.ToTensor()(img)
#outputs = model(inputs)


# In[14]:


#new_road_factory.classMap


# In[15]:


#class_label = "lane"
#classMap = new_road_factory.classMap
#threshold = 0
#showInferenceOnImage(img, outputs[0], class_label, threshold, classMap)


# In[16]:


#print("min", torch.min(outputs[0][0]), "max", torch.max(outputs[0][0]))

