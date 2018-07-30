
# coding: utf-8

# In[1]:


'''# Code to write files depending on the machine we are using
import os
from datetime import datetime
import logging

BASE_PATH = "/home/peo5032/Documents/COMP594"
MACHINE_NAME = os.uname()[1]
WRITE_ROOT = BASE_PATH + "/" + MACHINE_NAME
SESSION_INSTANCE = WRITE_ROOT + "/" + str(datetime.now())
# Logging setup and File
FILE_NAME = "run_information.log"
logging.basicConfig(filename=FILE_NAME, level=logging.DEBUG)

try:
    os.mkdir(BASE_PATH)
    logging.debug('successfully wrote ' + BASE_PATH)
except FileExistsError:
    logging.debug('we already have this path created!')
except OSError:
    logging.debug('something happened that it couldnt create this path?')

try:
    os.mkdir(WRITE_ROOT)
    logging.debug('successfully wrote ' +  WRITE_ROOT)
except FileExistsError:
    logging.debug('we already have this path created!')
except OSError:
    logging.debug('something happened that it couldnt create this path?') 
    
    
try:
    os.mkdir(SESSION_INSTANCE)
    logging.debug('successfully wrote' + SESSION_INSTANCE)
except FileExistsError:
    logging.debug('we already have this path created!')
except OSError:
    logging.debug('something happened that it couldnt create this path?') 
'''


# In[2]:


#read our excel file
import pandas as pd

df = pd.read_excel("./reference_code/Route_segments_urban_lanes_region.xlsx")


# In[3]:


#create our types
df.sort_values('URBAN')


# In[6]:


with open('./reference_code/urban_list.txt', 'w') as urban, open('./reference_code/non_urban_list.txt', 'w') as non_urban:
#route, begarm, endarm, urban, two lane, region, functional class
    for index, row in df.iterrows():
        tmp = str(row[0]) + '_' + str(int(row[1]))
        if row[3] == 1:
            for i in range(0,100):
                urban.write("cp /home/jjb24/wsdot/input/images/" + tmp + '.' + str(i) + '_b.png' + " /home/peo5032/Documents/COMP594/URBAN_IMAGES" + '\n')
                urban.write("cp /home/jjb24/wsdot/input/images/" + tmp + '.' + str(i) + '_i.png' + " /home/peo5032/Documents/COMP594/URBAN_IMAGES" + '\n')
                urban.write("cp /home/jjb24/wsdot/input/images/" + tmp + '.' + str(i) + '_d.png' + " /home/peo5032/Documents/COMP594/URBAN_IMAGES" + '\n')
        else:
            for i in range(0,100):
                non_urban.write("cp /home/jjb24/wsdot/input/images/" + tmp + '.' + str(i) + '_b.png'  + " /home/peo5032/Documents/COMP594/URBAN_IMAGES" + '\n')
                non_urban.write("cp /home/jjb24/wsdot/input/images/" + tmp + '.' + str(i) + '_i.png'  + " /home/peo5032/Documents/COMP594/URBAN_IMAGES" + '\n')
                non_urban.write("cp /home/jjb24/wsdot/input/images/" + tmp + '.' + str(i) + '_d.png'  + " /home/peo5032/Documents/COMP594/URBAN_IMAGES" + '\n')

        
    urban.close()
    non_urban.close()

