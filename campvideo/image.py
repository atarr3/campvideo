import cv2
import numpy as np
import os
import pandas as pd

from cv2 import imread,imwrite,polylines,imshow
from google.cloud import vision
from math import ceil
from os.path import splitext,basename,join
from shutil import copyfile

imtext_path = r'E:\Users\Alex\OneDrive\Documents\Research\campvideo\U2D Documents\2019\imagetext_errors.csv'
vid_dir = r'E:\Users\Alex\Desktop\ITH_Video'

# image class for text and face recognition
class Image(object):
    def __init__(self,im,name=None,ext=None):
        # BGR numpy array
        self.im = im
        # extension for image type, defaults to .png
        self.ext = '.png' if ext is None else ext
        # image properties
        if name is not None: self.name = name
        self.resolution = im.shape[:2]
    
    @classmethod
    def fromfile(cls,im_path):
        # read image and get filetype
        im = imread(im_path)
        name,ext = splitext(im_path)
        
        return cls(im,name=basename(name),ext=ext)
    
    # function for converting to byte string
    def tobytes(self):
        flag,enc = cv2.imencode(self.ext,self.im)
        
        if flag: 
            return enc.tobytes()
        else: 
            raise Exception('')
    
    
    # function for detecting and recognizing image text using Google Cloud API    
    def image_text(self,thr=0.035):
        # convert image to byte string
        content = self.tobytes()
        
        # google cloud vision client
        client = vision.ImageAnnotatorClient()
            
        # vision Image
        image = vision.types.Image(content=content)
            
        # detect text
        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        # plot image with bounding boxes
        copy = self.im.copy()
        for text in texts[1:]:
            vertices = np.array([(vertex.x, vertex.y)
                                 for vertex in text.bounding_poly.vertices],
                                dtype=np.int32)
            dx = vertices[0][0] - vertices[3][0]
            dy = vertices[0][1] - vertices[3][1]
            height = ceil(np.sqrt(dx ** 2 + dy ** 2))
            rel_height = height / self.resolution[0]
            
            if rel_height >= thr:
                bb = polylines(copy,[vertices],True,(0,255,0))
            else:
                bb = polylines(copy,[vertices],True,(0,0,255))
                
        imshow('f',bb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        out = join(r'E:\Users\Alex\Desktop\bbs',self.name+self.ext)
        imwrite(out,bb)
        
    data = pd.read_csv(imtext_path,index_col=0,usecols=[0,1,2])
    
    fpaths = [join(root,name) for root,dirs,files in os.walk(vid_dir)
                                  for name in files 
                                      if name.endswith(('.mp4','.wmv'))
                                      and basename(splitext(name)[0]) in data.index]
    # copy files into single directoryd
    os.mkdir(r'E:\Users\Alex\Desktop\imtext_vids')
    for fpath in fpaths:
        copyfile(fpath,join(r'E:\Users\Alex\Desktop\imtext_vids',basename(fpath)))
        
imtext_dir = r'E:\Users\Alex\Desktop\imtext'
fpaths = [join(root,name) for root,dirs,files in os.walk(imtext_dir)
                                  for name in files 
                                      if name.endswith(('.png'))]
    
for fpath in fpaths:
    im = Image.fromfile(fpath)
    im.image_text(0.035)
        