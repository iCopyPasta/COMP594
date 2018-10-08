
# coding: utf-8

# In[12]:


import warnings
import argparse
import DrawingWithTensors
import pandas as pd
from random import randint
import torch
import os
import argparse

# initiate the parser
parser = argparse.ArgumentParser(description = "List of options to run application when creating custom datset")

parser = argparse.ArgumentParser()  
parser.add_argument("-V", "--version", help="show program version", action="store_true")
parser.add_argument("-s", "--size", help="upper bound of number of images to include")
parser.add_argument("-t", "--rotations", help="degree of rotations in generation of images")
parser.add_argument("-r", "--root_folder", help="destination for root folder")
parser.add_argument("-i", "--iteration", help="which generation number we are using")
parser.add_argument("-d", "--dimensions", help="square image dimensions")

#defined defaults
upper_bound = 101
isize = 400
iteration = "1"
rotations="0"
ROOT = "/home/peo5032/Documents/COMP594/input/gen"


# read arguments from the command line
args = parser.parse_args()

# check for --version or -V
if args.version:  
    print("this is version 0.1", flush=True)
    
if args.size: 
    print("will create", upper_bound, "images", flush=True)
    upper_bound = int(args.size) + 1

if args.rotations: 
    print("rotations was", args.rotations, flush=True)
    
if args.root_folder:  
    os.makedirs(root_folder, exist_ok=True)
    print("destination was", args.root_folder, flush=True)
    
if args.iteration:
    print("iteration was", args.iteration, flush=True)
    iteration = args.iteration
    
if args.dimensions:
    print("dimension chosen was", args.dimensions,flush=True)
    isize = int(args.dimensions)
    
factor = 0.45

#TODO: ADD CODE TO MAKE DIRS WHEN THEY DO NOT EXIST FOR SOME MACHINE
os.makedirs(ROOT, exist_ok=True)
IMAGE_PATH = ROOT + iteration + "/roads"
os.makedirs(IMAGE_PATH, exist_ok=True)

TENSOR_PATH = ROOT + iteration + "/tensor_values"
os.makedirs(TENSOR_PATH, exist_ok=True)

PICKLE_PATH = ROOT + iteration

df = pd.DataFrame()
NumLanes = []
ShldrWidth = []
ShldrWidthCenter = []
RoadWidth = []
FileNames = []
imageGen = DrawingWithTensors.datasetFactory(IMAGE_SIZE = isize)
tmp_tensor = torch.zeros(7,isize,isize,dtype=torch.float32)


# In[ ]:


for i in range(0,upper_bound):
    if i % 10 == 0:
        print("Picture ",i, flush=True)
    c = randint(0,80)
    lanecount = randint(1,5)
    laneWidth = randint(17,35)
    lineWidth = randint(1,2)
    shoulderWidth = randint(0,89)
    
    #create tuple of information, img, and tensor
    tuple,img,tmp_tensor = imageGen.generateNewImageWithTensor(c,lanecount,laneWidth,lineWidth,shoulderWidth, tmp_tensor)
    roadWidth,laneCount,shoulderWidth,centerShldrWidth = tuple                 
    
    NumLanes.append(laneCount)
    ShldrWidth.append(shoulderWidth)
    RoadWidth.append (roadWidth)
    ShldrWidthCenter.append(centerShldrWidth)
    
    FileName = str(i) + ".png"
    FileNames.append(FileName)
    img.save(IMAGE_PATH + "/" + FileName,"PNG")
    
    #save tensor
    torch.save(tmp_tensor, TENSOR_PATH + "/"+ str(i) + '.pt')
    
    img.close()

df['NumLanes'] = NumLanes
df['ShldrWidth'] = ShldrWidth
df['RdwyWidth'] = RoadWidth
df['ShldrWidthCenter'] = ShldrWidthCenter
df['FileName'] = FileNames

df.to_pickle(PICKLE_PATH + "/train_images_v2.pkl")
print("program terminated", flush=True)

