import ImageGenerated_v2
import pandas as pd
from random import randint

factor = 0.45
PATH = "/home/peo5032/Documents/COMP594"
INPUT_PATH = PATH + "/input"


df = pd.DataFrame()
NumLanes = []
ShldrWidth = []
ShldrWidthCenter = []
RoadWidth = []
FileNames = []
imageGen = ImageGenerated_v2.roadImage2()
for i in range(0,100):
    if i % 10 == 0:
        print("Picture ",i)
    c = randint(0,80)
    lc = randint(1,5)
    laneWidth = randint(17,35)
    lineWidth = randint(1,2)
    shoulderWidth = randint(0,89)
    tuple,img = imageGen.generateImage(c,lc,laneWidth,lineWidth,shoulderWidth)
    roadWidth,laneCount,shoulderWidth,centerShldrWidth = tuple                 
    NumLanes.append(laneCount)
    ShldrWidth.append(shoulderWidth)
    RoadWidth.append (roadWidth)
    ShldrWidthCenter.append(centerShldrWidth)
    FileName = str(i) + ".png"
    FileNames.append(FileName)
    img.save(INPUT_PATH + "/" + FileName,"PNG")

df['NumLanes'] = NumLanes
df['ShldrWidth'] = ShldrWidth
df['RdwyWidth'] = RoadWidth
df['ShldrWidthCenter'] = ShldrWidthCenter
df['FileName'] = FileNames

df.to_pickle(INPUT_PATH + "/train_images_v2.pkl")
