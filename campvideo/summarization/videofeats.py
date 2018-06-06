import cv2
import numpy as np

from functools import partial
from queue import Queue
from threading import Thread

class VideoFeats:
    def __init__(self,vid_fpath,qsize=512):
        self.__cap = cv2.VideoCapture(vid_fpath)
        self.__buffer = Queue(maxsize=qsize)
        self.frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.resolution = (int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                           
        # # start filling buffer with frames on separate thread
        t = Thread(target=self.__fill,daemon=True)
        t.start()
        
        # compute features
        hog = HOG(self.resolution)
        labhist = Histogram(self.resolution)
        
        self.labhist = np.zeros((self.frame_count,labhist.dimension),dtype=np.float32)
        self.hog = np.zeros((self.frame_count,hog.dimension),dtype=float)
        
        for i in range(self.frame_count):
            cf = self.__buffer.get()
            self.labhist[i,...] = labhist.compute(cf)
            self.hog[i,...] = hog.compute(cf)
        
    def __stop(self):
        self.__cap.release()
        
    # function for filling buffer with specified frames
    def __fill(self):
        while True:
            flag,frame = self.__cap.read()       
            if not flag: self.__stop(); return
        
            # put frame in buffer
            self.__buffer.put(frame)
            
class Histogram:
    def __init__(self,size,nbins=23):
        self.dimension = nbins ** 3
        self.numel = np.prod(size)
        self.__hist = partial(cv2.calcHist,channels=[0,1,2],mask=None,
                              histSize=[nbins,]*3,ranges=[0,255]*3)
    
    def compute(self,frame):
        counts = self.__hist([cv2.cvtColor(frame,cv2.COLOR_BGR2Lab)])
        return counts.flatten() / self.numel
        
class HOG:
    def __init__(self,size,nbins=9,block_size=(16,16),cell_size=(8,8)):
        self.numel = np.prod(size)
        self.__detector = cv2.HOGDescriptor(_winSize=size,_blockSize=block_size,
                                            _blockStride=(8,8),_cellSize=cell_size,
                                            _nbins=nbins)
        self.dimension = self.__detector.getDescriptorSize()
                                            
    def compute(self,frame):
        return self.__detector.compute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)).flatten()
            

