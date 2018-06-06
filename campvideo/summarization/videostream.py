import cv2
# import keras
import numpy as np
import os
import sys

# remove this in final release (python 3 only)
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue
from threading import Thread

class VideoStream:
    def __init__(self,fpath,queue_size=128):
        # check that file exists
        if not os.path.exists(fpath):
            raise IOError("file '{0}' does not exist".format(fpath))
            
        # create frame buffer
        self.__buffer = Queue(maxsize=queue_size)
        # create VideoCapture object
        self.__stream = cv2.VideoCapture(fpath)
        # video properties
        self.file = fpath
        self.title = os.path.splitext(os.path.basename(fpath))[0]
        self.frame_count = int(self.__stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.__stream.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps
        self.resolution = (int(self.__stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.__stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # release video capture
        self.__release()
                           
    # function for filling buffer with specified frames
    def __fill(self,inds):
        # empty buffer
        self.__buffer.queue.clear()
        # fill buffer
        for ind in inds:
            # retrieve current frame
            self.__stream.set(1,ind)
            __,cf = self.__stream.read()
            # put in queue
            self.__buffer.put(cf)
    
    # close VideoCapture        
    def __release(self):
        self.__stream.release()
        
    # method for grabbing specified frames
    def frames(self,frame_inds=None ,size=None,colorspace='BGR'):
        """Returns the requested frames as an N x H x W x C uint8 numpy array.
        
        Args:
            frame_ind:  The indices of the frames to be retrieved
            size:       Two-element tuple specifying the desired size of the 
                        frames with width as the first element and height as the 
                        second element. Defaults to the native resolution.
            colorspace: String specifying the colorspace representation of the 
                        frames. Valid options include 'RGB','BGR','YUV','HSV',
                        'Lab', and 'gray'. Defaults to 'BGR'. 
        """
        # color conversion dictionary
        colors = {'RGB':cv2.COLOR_BGR2RGB,'BGR':None,'YUV':cv2.COLOR_BGR2YUV,
                  'HSV':cv2.COLOR_BGR2HSV,'Lab':cv2.COLOR_BGR2Lab,
                  'gray':cv2.COLOR_BGR2GRAY}
                  
        # if no frame indices specified, return all frames
        if frame_inds is None or frame_inds == []:
            frame_inds = range(self.frame_count)
        # create iterable if one index specified
        if not np.iterable(frame_inds):
            frame_inds = [frame_inds]
        # desired frame resolution
        if size is None:
            size = self.resolution
            
        # check if frame_ind contains out-of-bounds entries
        if np.max(frame_inds) >= self.frame_count or np.min(frame_inds) < 0:
            raise IndexError("frame indices out of bounds. Please specify " 
                              "indices from 0 to {0}"
                              .format(self.frame_count-1))
        # check that specified colorspace is correct
        if colorspace not in colors: 
            raise ValueError("{0} is not a valid colorspace. Please see "
                              "function details for a list of valid colorspaces"
                              .format(colorspace))
        # open up stream as cv2.VideoCapture object
        self.__stream = cv2.VideoCapture(self.file)                      
        # start thread for filling frame buffer
        t = Thread(target=self.__fill, args=(frame_inds,))
        t.daemon = True
        t.start()
        
        # number of frames and layers
        n = np.size(frame_inds)
        if colorspace == 'gray':
            frames = np.empty((n,size[1],size[0]),dtype='uint8')
        else:
            frames = np.empty((n,size[1],size[0],3),dtype='uint8')
            
        # read frames from queue
        for i in range(n):
            cf = self.__buffer.get()
            # resize
            if (cf.shape[1],cf.shape[0]) != size:
                cf = cv2.resize(cf,size)
            # convert colorspace
            if colors[colorspace]:
                cf = cv2.cvtColor(cf,colors[colorspace])
            
            frames[i,...] = cf
            
        # close thread
        t.join()
        # release VideoCapture
        self.__release()
            
        return frames
        
    # method for extracting fc-6 features
    def feats(frame_inds=None):
        """Returns the fc-6 features for the requested frames as a numpy array.
        
        Args:
            frame_ind: The indices of the frames to compute the features from
        """