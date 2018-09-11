from PIL import Image
from random import randint
import numpy as np
import pandas as pd
import sys
PATH = "/home/peo5032/Documents/COMP594"
INPUT_PATH = PATH + "/input"

class roadGenerator(object):

    def __init__(self):
        
        try:
            self.IMAGE_SIZE=400
            df = pd.read_pickle(INPUT_PATH + "/train_images_v2.pkl")
            df = df[df.MedianWidth > 0]
            self.shldrCenterStdDev = df['ShldrWidthCenter'].std()
            self.shldrCenterMean = df['ShldrWidthCenter'].mean()
            self.RdwyWidthStdDev = df['RdwyWidth'].std()
            self.RdwyWidthMean = df['RdwyWidth'].mean()
            self.NumLanesStdDev = df['NumLanes'].std()
            self.NumLanesMean = df['NumLanes'].mean()
            self.ShldrWidthStdDev = df['ShldrWidth'].std()
            self.ShldrWidthMean = df['ShldrWidth'].mean()
        except IOError:
            print('An error occured trying to read the file.')
            
    def drawBackground(self,imgMap):
        choice = randint(0,2)
        if choice == 0:
            for i in range(self.IMAGE_SIZE):
                for j in range(self.IMAGE_SIZE):
                    imgMap[i,j] = (randint(0,255),randint(0,255),randint(0,255))
        elif choice == 1:
            rgb = (randint(0,255),randint(0,255),randint(0,255))
            for i in range(self.IMAGE_SIZE):
                for j in range(self.IMAGE_SIZE):
                    imgMap[i,j] = rgb
        elif choice == 2:
            r = randint(0,255)
            rD = randint(0,40)
            g = randint(0,255)
            gD = randint(0,40)
            b = randint(0,255)
            bD = randint(0,40)
            for i in range(self.IMAGE_SIZE):
                for j in range(self.IMAGE_SIZE):
                    imgMap[i,j] = (max(0,min(255,int(np.random.normal(r,rD)))),
                                   max(0,min(255,int(np.random.normal(g,gD)))),
                                   max(0,min(255,int(np.random.normal(b,bD)))))
        else:
            print("bug in drawBackground")
            sys.exit(-1)
           
    def drawLine(self,imgMap,start,width,red,redDev,green,greenDev,blue,blueDev,onLen,offLen):
        if start < 0 or start + width >= self.IMAGE_SIZE:
            print(start,width)
        for i in range(start,start+width):
           on = True
           dist = onLen
           for j in range(self.IMAGE_SIZE):
                if on == True:
                    r = max(0,min(255,int(np.random.normal(red,redDev))))
                    g = max(0,min(255,int(np.random.normal(green,greenDev))))
                    b = max(0,min(255,int(np.random.normal(blue,blueDev))))
                    imgMap[i,j] = (r,g,b)
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
     
    def generateImage(self,centerShldrWidth,laneCount,laneWidth,lineWidth,shoulderWidth):
        img = Image.new('RGB',(self.IMAGE_SIZE,self.IMAGE_SIZE))
        imgMap = img.load()
        
        self.drawBackground(imgMap)
        
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

        self.drawLine(imgMap,start,centerShldrWidth,128,20,128,20,128,20,0,0)
        start += centerShldrWidth
        
        for i in range(laneCount):
            if i == 0:
                self.drawLine(imgMap,start,lineWidth,200,40,200,40,50,40,0,0)
            else:
                self.drawLine(imgMap,start,lineWidth,200,40,200,40,200,40,20,20)
            start += lineWidth 
            self.drawLine(imgMap,start,laneWidth,128,40,128,40,128,40,0,0)
            start += laneWidth - lineWidth 
        
        self.drawLine(imgMap,start,lineWidth,200,40,400,40,200,40,0,0)
        start += lineWidth
        self.drawLine(imgMap,start,shoulderWidth, 128,40,128,40,128,40,0,0)
        
        #roadWidth = centerShldrWidth + laneCount*laneWidth + shoulderWidth
        roadWidth = laneCount*laneWidth

        #roadWidth = (roadWidth*factor - self.RdwyWidthMean)/self.RdwyWidthStdDev
        roadWidth = roadWidth*factor
        #laneCount = (laneCount - self.NumLanesMean)/self.NumLanesStdDev
        #shoulderWidth = (shoulderWidth*factor - self.shldrCenterMean)/self.shldrCenterStdDev
        shoulderWidth = shoulderWidth*factor
        #centerShldrWidth = (centerShldrWidth*factor - self.ShldrWidthMean)/self.ShldrWidthStdDev
        centerShldrWidth = centerShldrWidth*factor

        return (roadWidth,laneCount,shoulderWidth,centerShldrWidth),img