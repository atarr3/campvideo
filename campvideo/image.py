import cv2
#import face_recognition
import numpy as np

from bisect import bisect
from cv2 import imread,polylines,imshow
from google.api_core.exceptions import DeadlineExceeded
from google.cloud import vision
from math import ceil
from os.path import splitext,basename,exists
from pkg_resources import resource_filename

MODEL_PATH = resource_filename('campvideo','models/frozen_east_text_detection.pb')

# load neural net for text bounding box detection
if exists(MODEL_PATH):
    text_net = cv2.dnn.readNet(MODEL_PATH)
else:
    raise Exception('Model data not installed. Please install the models by '
                    'typing the following command into the command line:\n\n'
                    'download_models')

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
    
    # show image
    def show(self):
        cv2.imshow('image',self.im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # function for converting to byte string
    def tobytes(self):
        flag,enc = cv2.imencode(self.ext,self.im)
        
        if flag: 
            return enc.tobytes()
        else: 
            raise Exception('Unable to encode image')
            
    # function for detecting and recognizing faces in an image
    def image_faces(self):
        pass
    
    # function for detecting and recognizing image text using Google Cloud API    
    def image_text(self,bb_thr=0.035,bb_count=25,plot_image=False):
        """ Detect and recognize artificial or scene text in the image
        
        Args:
            bb_thr : float, optional       
                Minimum relative height of detected bounded for text to be 
                kept. Default value is 0.035 (3.5% of image height).
            bb_count : int, optional     
                Maximum number of words to return. Keeps the `count` largest 
                (by height) bounding boxes. Default value is 25.
            plot_image : bool, optional
                Flag for plotting image with detected text bounding boxes 
                overlaid. Default value is False.
        """
        # checks of Google API has already been called for this image
        if not hasattr(self,'_texts'):
            # checks if any text is in the image before GCP API call
            if not self._has_text():
                # function will return [] on subsequent callse
                self._texts = []
                return []
                
            # convert image to byte string
            content = self.tobytes()
            
            # google cloud vision client
            client = vision.ImageAnnotatorClient()
                
            # vision Image
            image = vision.types.Image(content=content)
                
            # detect text
            try:
                response = client.text_detection(image=image,timeout=10)
            except DeadlineExceeded:
                # timed out, try again
                response = client.text_detection(image=image,timeout=10)
                
            # store results for future use
            self._texts = response.text_annotations
        
        # plot image with bounding boxes
        if plot_image:
            copy = self.im.copy()
            
        # output list of text
        texts = self._texts[1:] # first entry is all text in the image
        rel_heights = []
        inds = []
        for ind,text in enumerate(texts):
            vertices = np.array([(vertex.x, vertex.y)
                                 for vertex in text.bounding_poly.vertices],
                                dtype=np.int32) # polylines requires int32
            # get rotated bb height (assumes rotation is less than 90 degrees)
            dx = vertices[0][0] - vertices[3][0]
            dy = vertices[0][1] - vertices[3][1]
            height = ceil(np.sqrt(dx ** 2 + dy ** 2))
            rel_height = height / self.resolution[0]

            # filter bb based on relative height and total bb count            
            if rel_height >= bb_thr:
                # insertion index that sorts based on asecending height
                ins = bisect(rel_heights,rel_height)
                # sorted heights and corresponding index in self._texts[1:]
                rel_heights.insert(ins,rel_height)
                inds.insert(ins,ind)
                if plot_image:
                    bb = polylines(copy,[vertices],True,(36,255,12))
            else:
                if plot_image:
                    bb = polylines(copy,[vertices],True,(0,0,255))
        
        if plot_image:        
            imshow('image with bounding boxes',bb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # return `count` largest texts, in order as they were detected in the
        # image
        return [texts[i].description for i in sorted(inds[-bb_count:])]
    
    # function for checking if text is contained in the image using the EAST
    # text detector from cv2. The purpose of this function is to avoid making
    # API calls to the GCP when no text is present on the image. 
    def _has_text(self,score_thr=0.5):
        # resize image to nearest dimensions that are a multiple of 32
        h,w = self.resolution
        # resize dimensions, detection is somewhat sensitive to this
        ar = w / h # aspect ratio
        new_w = min(480,w) // 32 * 32
        # nearest (in aspect ratio) height that is a multiple of 32
        new_h = int(np.round(int(new_w/ar) / 32)) * 32
        # make copy to preserve original image
        copy = self.im.copy()
        copy = cv2.resize(copy,(new_w,new_h))
        
        # relevant layer names
        layers = ["feature_fusion/Conv_7/Sigmoid"]
            
        # construct blob and compute network outputs at `layers`
        blob = cv2.dnn.blobFromImage(copy,1.0,(new_w,new_h),
                    (123.68,116.78,103.94),swapRB=True,crop=False)
        text_net.setInput(blob)
        scores = text_net.forward(layers)[0]
        
        # check if any bounding box scores exceed confidence threshold
        return np.any(scores[0,0] > score_thr)
    
# function for performing image text detection over a batch of images
def batch_image_text(ims,bb_thr=0.035,count=25):
    """ Detect and recognize artificial or scene text in an array of images
    
    Args:
        ims : list
            List of 
        bb_thr : float, optional       
            Minimum relative height of detected bounded for text to be 
            kept. Default value is 0.035 (3.5% of image height).
        bb_count : int, optional     
            Maximum number of words to return. Keeps the `count` largest 
            (by height) bounding boxes. Default value is 25.
    """
    pass
    