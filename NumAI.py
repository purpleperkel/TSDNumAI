from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import *
from skimage.transform import resize
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os
import ReadImageTester

path = "TSDImageMerge/"
dirs = os.listdir( path )
final_size = 180

def get_preds():
    for sdir in dirs:
         if sdir == '.DS_Store':
             continue
         print("Dir Path: "+path+sdir+"\n")
         if os.path.isdir(path+sdir):
                 for item in os.listdir(path+sdir):
                    if item == '.DS_Store':
                        continue
                    #else:
                    #    print("Item found!\n")
                    #print(item)
                    filePath = path+sdir+"/"+item
                    if os.path.isfile(filePath):
                        print("Image Path: "+filePath+"\n")
                        #im = Image.open(filePath)
                        pred = ReadImageTester.get_pred(filePath, False) 
                        if (pred is not None):
                          print(pred)
                    else:
                        print("Error path: " + filePath)


get_preds()