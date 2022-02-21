#from skimage.transform import resize
#from tensorflow import keras
#from keras.models import Sequential
#from keras import layers

#data_augmentation = keras.Sequential(
#    [
#        layers.RandomFlip("horizontal"),
#        layers.RandomRotation(0.1),
#    ]
#)



from typing import MappingView
from PIL import Image
import os, sys
import random
from datetime import datetime

path = "TSDImageMerge/"
dirs = os.listdir( path )
final_size = 180

def resize_aspect_fit():
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
                        im = Image.open(filePath)
                        f, e = os.path.splitext(filePath)
                        size = im.size
                        ratio = float(final_size) / max(size)
                        new_image_size = tuple([int(x*ratio) for x in size])
                        im = im.resize(new_image_size, Image.ANTIALIAS)
                        new_im = Image.new("RGB", (final_size, final_size))
                        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
                        new_im = new_im.convert("L")
                        new_im.save(f + 'resized.jpg', 'JPEG', quality=90)
                        #""""
                        for i in range(0,50):
                            random.seed(datetime.now())
                            randAngle = random.random()*15
                            rotateImage = None
                            if (i % 2 == 0):
                                rotateImage = new_im.rotate(randAngle,resample=Image.BICUBIC)
                                rotateImage.save(f + "rotate"+("{:.2f}".format(randAngle))+".jpg", 'JPEG', quality=90)
                            else:
                                rotateImage = new_im.rotate(-randAngle,  resample=Image.BICUBIC)
                                rotateImage.save(f + "rotate"+("{:.2f}".format(-randAngle))+".jpg", 'JPEG', quality=90)
                        #"""

                        os.remove(filePath)
                    else:
                        print("Error path: " + filePath)


resize_aspect_fit()