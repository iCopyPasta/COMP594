
# coding: utf-8

# In[6]:


# Code to write files depending on the machine we are using
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

