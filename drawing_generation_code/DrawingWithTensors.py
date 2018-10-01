
# coding: utf-8

# In[1]:


from PIL import Image
from random import randint
import numpy as np
import pandas as pd
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import torchvision
get_ipython().run_line_magic('matplotlib', 'inline')

PATH = "/home/peo5032/Documents/COMP594"
INPUT_PATH = PATH + "/input"

class datasetFactory(object):

    def __init__(self, IMAGE_SIZE = 400, listOfClasses=["background","left-shoulder","left-yellow-line-marker","white-lane-markers","lane",
                         "right-white-line-marker", "right-shoulder"]):
        
        try:
            self.IMAGE_SIZE=IMAGE_SIZE
            self.classMap = dict()
            
            if(len(listOfClasses) <= 0):
                self.classList = ["background","left-shoulder","left-yellow-line-marker","white-lane-markers","lane",
                         "right-white-line-marker", "right-shoulder"]
            else: 
                self.classList = listOfClasses
            
            for i in range(0, len(self.classList)):
                self.classMap[self.classList[i]] = i
                               
            self.NUM_CLASSES = len(self.classList)
            self.tensorMap = torch.zeros([self.NUM_CLASSES,
                                          self.IMAGE_SIZE,
                                          self.IMAGE_SIZE],
                                         dtype = torch.int64)
            
            
        except IOError:
            print('An error occured trying to read the file.')
            
    
    def drawBackground(self, imgMap):
        
        # choose one color for background
        rgb = (randint(0,255),randint(0,255),randint(0,255))
        
        for i in range(self.IMAGE_SIZE):
            for j in range(self.IMAGE_SIZE):
                #DRAW BACKGROUND
                imgMap[i,j] = rgb
        
    def computeTensorBackground(self, imgMap):
        # background channel
        self.tensorMap[0] = torch.zeros([self.IMAGE_SIZE, self.IMAGE_SIZE])
        
        # in the background channel, update values to know which are being used in other tensors
        for i in range(1,len(self.classList)):
            self.tensorMap[0][self.tensorMap[i] == 1] = -1 
                
        # flush to allow background only values
        # UNUSED CLASS LABEL: 999
        self.tensorMap[0][self.tensorMap[0] == 0] = 1
        self.tensorMap[0][self.tensorMap[0] == -1] = 0
        
                  
    def drawStraightLine(self,imgMap,start,width,red,redDev,green,greenDev,blue,blueDev,onLen,offLen,class_type_flag):
        if start < 0 or start + width >= self.IMAGE_SIZE:
            print(start,width, "ERROR")
            exit()
            
        class_type_corresponding_channel = self.classMap[class_type_flag]
        #print("DRAW CLASS", class_type_flag)
        
        for i in range(start,start+width):
            on = True
            dist = onLen
            for j in range(self.IMAGE_SIZE):
                if on == True:
                    r = max(0,min(255,int(np.random.normal(red,redDev))))
                    g = max(0,min(255,int(np.random.normal(green,greenDev))))
                    b = max(0,min(255,int(np.random.normal(blue,blueDev))))
                    imgMap[i,j] = (r,g,b)
                    self.tensorMap[class_type_corresponding_channel, i,j] = 1
                    if onLen > 0:
                        dist = dist - 1
                        if dist < 0:
                           dist = offLen
                           on = not on
                else:
                    dist = dist - 1
                    if dist < 0:
                        dist = onLen
                        on = not on
                        
    def drawWhiteLaneDevisor(self,imgMap,start,width,red,redDev,green,greenDev,blue,blueDev,onLen,offLen,class_type_flag):
        if start < 0 or start + width >= self.IMAGE_SIZE:
            print(start,width, "ERROR")
            exit()
            
        class_type_corresponding_channel = self.classMap[class_type_flag]
        
        for i in range(start,start+width):
            on = True
            dist = onLen
            for j in range(self.IMAGE_SIZE):
                if on == True:
                    r = max(0,min(255,int(np.random.normal(red,redDev))))
                    g = max(0,min(255,int(np.random.normal(green,greenDev))))
                    b = max(0,min(255,int(np.random.normal(blue,blueDev))))
                    imgMap[i,j] = (r,g,b)
                    self.tensorMap[class_type_corresponding_channel, i,j] = 1
                    if onLen > 0:
                        dist = dist - 1
                        if dist < 0:
                           dist = offLen
                           on = not on
                else:
                    #fill in the image with grey
                    r = max(0,min(255,int(np.random.normal(128,40))))
                    g = max(0,min(255,int(np.random.normal(128,40))))
                    b = max(0,min(255,int(np.random.normal(128,40))))
                    
                    imgMap[i,j] = (r,g,b)
                    self.tensorMap[class_type_corresponding_channel, i,j] = 1
                    dist = dist - 1
                    if dist < 0:
                        dist = onLen
                        on = not on                    
                        
                
    def generateNewImageWithTensor(self,centerShldrWidth,laneCount,laneWidth,lineWidth,shoulderWidth):
        img = Image.new('RGB',(self.IMAGE_SIZE,self.IMAGE_SIZE))
        imgMap = img.load()
        
        factor = 0.45   # ft/px
        # 0 to 36
        #centerShldrWidth=randint(0,80)
        # 0 to 5
        #laneCount = randint(0,5)
        # 8 to 15
        #laneWidth = randint(17,34) 
        # 4 to 6 in
        #lineWidth = randint(1,2)
        # 8 to 40
        #shoulderWidth=randint(0,89)
        start = (self.IMAGE_SIZE - centerShldrWidth - (laneCount+1)*lineWidth - laneCount * laneWidth - shoulderWidth)//2
        
        if start < 10:
           print(centerShldrWidth,laneCount,laneWidth,lineWidth,shoulderWidth,"EXCEEDED IMAGE_SIZE")
           sys.exit(-1)
            
        #DRAW BACKGROUND
        self.drawBackground(imgMap)

        #DRAW: left shoulder    
        self.drawStraightLine(imgMap,start,centerShldrWidth,128,20,128,20,128,20,0,0, "left-shoulder")
        
        # move pointer by the shoulder width
        start += centerShldrWidth
        #print("laneCount is:",laneCount)
        
        # for the number of lanes we have, draw them
        for i in range(laneCount):
            #print("printing lane number:",i)
            if i == 0:
                #DRAW left-yellow-line-marker
                self.drawStraightLine(imgMap,start,lineWidth,200,40,200,40,50,40,0,0, "left-yellow-line-marker")
            else:
                #DRAW white-lane-marker
                self.drawWhiteLaneDevisor(imgMap,start,lineWidth,200,40,200,40,200,40,20,20, "white-lane-markers")
            
            #move over a white-lane-markers line
            start += lineWidth 
            
            #DRAW our lane
            self.drawStraightLine(imgMap,start,laneWidth-lineWidth,128,40,128,40,128,40,0,0, "lane")
            
            #move pointer by the lane width
            start += laneWidth - lineWidth 
        
        #DRAW white-line-marker
        #self.drawStraightLine(imgMap,start,lineWidth,200,40,400,40,200,40,0,0, "right-white-line-marker")
        self.drawStraightLine(imgMap,start,lineWidth,255,25,255,25,255,25,0,0, "right-white-line-marker")
        
        #move pointer by the white-line width
        start += lineWidth
        
        #DRAW right-shoulder
        self.drawStraightLine(imgMap,start,shoulderWidth, 128,40,128,40,128,40,0,0, "right-shoulder")
        
        #fill in background tensor
        self.computeTensorBackground(imgMap)
        
        #roadWidth = centerShldrWidth + laneCount*laneWidth + shoulderWidth
        roadWidth = laneCount*laneWidth

        #roadWidth = (roadWidth*factor - self.RdwyWidthMean)/self.RdwyWidthStdDev
        roadWidth = roadWidth*factor
        #laneCount = (laneCount - self.NumLanesMean)/self.NumLanesStdDev
        #shoulderWidth = (shoulderWidth*factor - self.shldrCenterMean)/self.shldrCenterStdDev
        shoulderWidth = shoulderWidth*factor
        #centerShldrWidth = (centerShldrWidth*factor - self.ShldrWidthMean)/self.ShldrWidthStdDev
        centerShldrWidth = centerShldrWidth*factor

        return (roadWidth,laneCount,shoulderWidth,centerShldrWidth),img, self.tensorMap    
    
    def showClassLabaelOnImage(self, img, tensor, class_label):
        imgTMP = img.copy()
        imgMap = imgTMP.load()
        class_type_corresponding_channel = self.classMap[class_label]
        
        for i in range(0, self.IMAGE_SIZE):
            for j in range(0, self.IMAGE_SIZE):
                if tensor[class_type_corresponding_channel, i,j] == 1:
                    #show class label in black
                    imgMap[i,j] = (0,0,0)
                #else:
                #    imgMap[i,j] = (0,0,0)
        
        return imgTMP
