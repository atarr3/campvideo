from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os

from functools import partial
from queue import Queue
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread

class Video:
    def __init__(self,fpath,queue_size=512):
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
        
        # close stream
        self.__release()
                           
    # function for filling buffer with specified frames
    def __fill(self,inds=None):
        # open up stream as cv2.VideoCapture object
        self.__stream = cv2.VideoCapture(self.file)
        # empty buffer
        self.__buffer.queue.clear()
        
        # fill buffer
        if inds is None: # grab all frames
            while True:
                flag,frame = self.__stream.read()       
                if not flag: self.__release(); return
                self.__buffer.put(frame)
        else:
            for ind in inds:
                # set stream to current frame
                self.__stream.set(1,ind)
                __,frame = self.__stream.read()
                self.__buffer.put(frame)
                
        # close stream
        self.__release()
    
    # close VideoCapture stream        
    def __release(self):
        self.__stream.release()
        
    # method for grabbing specified frames
    def frames(self,frame_inds=None,size=None,colorspace='BGR'):
        """Returns the requested frames as an N x H x W x C uint8 numpy array.
        
        Args:
            frame_ind:  The indices of the frames to be retrieved
            size:       Two-element tuple specifying the desired size of the 
                        frames with width as the first element and height as 
                        the second element. Defaults to the native resolution.
            colorspace: String specifying the colorspace representation of the 
                        frames. Valid options include 'RGB','BGR','YUV','HSV',
                        'Lab', and 'gray'. Defaults to 'BGR'. 
        """
        # color conversion dictionary
        colors = {'RGB' : cv2.COLOR_BGR2RGB,'BGR' : None,'YUV' : cv2.COLOR_BGR2YUV,
                  'HSV' : cv2.COLOR_BGR2HSV,'Lab' : cv2.COLOR_BGR2Lab,
                  'gray' : cv2.COLOR_BGR2GRAY}
        
        # create iterable if one index specified
        if frame_inds is not None and not np.iterable(frame_inds):
            frame_inds = [frame_inds]
        if frame_inds is not None and len(frame_inds) == 0:
            return []
        # frame resolution
        if size is None:
            size = self.resolution
            
        # check if frame_ind contains out-of-bounds entries
        if frame_inds is not None:
            if np.max(frame_inds) >= self.frame_count or np.min(frame_inds) < 0:
                raise IndexError("frame indices out of bounds. Please specify " 
                                 "indices from 0 to {0}"
                                  .format(self.frame_count-1))
            # check that specified colorspace is correct
            if colorspace not in colors: 
                raise ValueError("{0} is not a valid colorspace. Please see "
                                  "function details for a list of valid colorspaces"
                                  .format(colorspace))
        
        # start thread for filling frame buffer
        t = Thread(target=self.__fill,args=(frame_inds,),daemon=True)
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
            if colorspace != 'BGR':
                cf = cv2.cvtColor(cf,colors[colorspace])
            
            frames[i] = cf
            
        # close thread
        t.join()
            
        return frames
    
    # method for computing frame features
    def feats(self,dsf=1):
        """Computes the Lab histogram and HOG features for the frames in the 
           video. Returns a tuple of feature arrays (labhist, hog)
        
        Args:
            dsf: Downsampling factor. The features are computed for frames 
                 np.arange(0,self.frame_count,dsf). Default is 1.
        """
        
        # frame indices
        frame_inds = np.arange(0,self.frame_count,dsf)
        # start filling buffer with frames on separate thread
        t = Thread(target=self.__fill,args=(frame_inds,),daemon=True)
        t.start()
        
        # feature class instantiation
        hog = HOG(self.resolution)
        labhist = LabHistogram(self.resolution)
        
        # number of frames after downsampling
        n = len(frame_inds)
        
        labhist_feat = np.zeros((n,labhist.dimension),dtype=np.float32)
        hog_feat = np.zeros((n,hog.dimension),dtype=float)
        
        for i in range(n):
            cf = self.__buffer.get()
            labhist_feat[i] = labhist.compute(cf)
            hog_feat[i] = hog.compute(cf)
    
        self.__buffer.all_tasks_done
        
        return (labhist_feat, hog_feat)
    
    # function for adaptively selecting keyframes via submodular optimization
    def kf_adaptive(self,l1=1,l2=5,niter=10,dsf=1):
        """Generates anarray of keyframes using an adaptive keyframe selection
           algorithm. 
        
        Args:
            l1:  Penalty for representativeness. Default is 1
            l2:  Penalty for summary length. Default is 5
            dsf: Downsampling factor for thinning frames before running the
                 algorithm. Increasing this will improve runtime. Default is 1
                 (no downsampling)
        """
        # prevents misuse
        dsf = int(dsf)
        
        # get histogram and HOG features
        feats_l, feats_h = self.feats(dsf=dsf)
        
        # video properties
        n = feats_h.shape[0]
            
        # pairwise comparisons of features
        cfunc = partial(cv2.compareHist,method=cv2.HISTCMP_CHISQR_ALT)
        
        w = cosine_similarity(feats_h)
        d = 0.25 * pairwise_distances(feats_l,metric=cfunc,n_jobs=-1)
        
        # computes objective function value for a given index set
        def objective(kf_ind):
            n_summ = len(kf_ind)
        
            # representativeness
            r = w[kf_ind].max(axis=0).sum()
            # uniqueness
            d_sub = d[np.ix_(kf_ind,kf_ind)]
            u = 0
            for j in range(1,len(d_sub)):
                # minumum distance to all previously added frames
                u += d_sub[j,:j].min()
                
            return r + l1*u + l2*(n-n_summ)
            
        # keyframe index set
        best_obj = 0
        
        # submodular optimization
        for _ in range(niter):       
            # random permutation of frame indices
            u = np.random.permutation(n)
        
            # initial keyframe index sets
            X = np.empty(0,dtype=int)
            Y = np.arange(n)
            
            # initial distances (similarity)
            wX_max = np.zeros(n,dtype=float)
            wY_max = w.max(axis=1)
        
            for uk in u:
                # Y index set with uk removed
                Y_t = Y[Y != uk]
                
                # update maximum similarity between each i and Y \ uk
                wY_max_t = wY_max.copy()
                change_ind = w[uk] >= wY_max
                if np.any(change_ind):
                    wY_max_t[change_ind] = w[np.ix_(change_ind,Y_t)].max(axis=1)
                
                # minimum distance from uk to X (uniqueness)
                dX_min = d[uk,X].min() if len(X) > 0 else 1.0
                # minimum distance from uk to Y \ uk
                dY_min = d[uk,Y_t].min() if len(Y_t) > 0 else 1.0
                    
                # change in objective function for X set
                df_X =   np.maximum(0,w[uk] - wX_max).sum() + l1*dX_min - l2
                # change in objective function for Y set
                df_Y = -(np.maximum(0,w[uk] - wY_max_t).sum() + l1*dY_min - l2)
            
                # probabilistically add/remove uk
                a = max(df_X,0.0)
                b = max(df_Y,0.0)      
                if a + b == 0.0:
                    prob_X = 0.5
                else:
                    prob_X = a / (a + b)
                    
                # add to X, no change in Y
                if np.random.uniform() < prob_X:
                    X = np.insert(X,np.searchsorted(X,uk),uk)
                    # update maximum similarity between each i and current X
                    wX_max = np.maximum(wX_max,w[uk])
                # remove from Y, no change in X
                else:
                    Y = Y_t
                    # update maximum similarity between each i and Y \ uk
                    wY_max = wY_max_t
                    
            # update keyframe set
            obj = objective(X)
            if obj > best_obj:
                best_kf_ind = X.copy()
                best_obj = obj
        
        # convert downsampled frame indices to original scale
        return dsf * best_kf_ind
        
            
class LabHistogram:
    def __init__(self,size,nbins=23):
        self.dimension = nbins ** 3
        self.numel = np.prod(size)
        self.__hist = partial(cv2.calcHist,channels=[0,1,2],mask=None,
                              histSize=[nbins,]*3,ranges=[0,256]*3)
    
    def compute(self,frame):
        counts = self.__hist([cv2.cvtColor(frame,cv2.COLOR_BGR2Lab)])
        return cv2.normalize(counts,counts,norm_type=cv2.NORM_L1).flatten()
        
class HOG:
    def __init__(self,size,nbins=9,block_size=(16,16),cell_size=(8,8)):
        self.numel = np.prod(size)
        self.__detector = cv2.HOGDescriptor(_winSize=size,_blockSize=block_size,
                                            _blockStride=(8,8),_cellSize=cell_size,
                                            _nbins=nbins)
        self.dimension = self.__detector.getDescriptorSize()
                                            
    def compute(self,frame):
        return self.__detector.compute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)).flatten()
