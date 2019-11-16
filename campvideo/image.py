import cv2
#import face_recognition
import numpy as np
import os
import warnings

from bisect import bisect
from campvideo import Video
from cv2 import imread,polylines,imshow
from google.cloud import vision
from math import ceil
from os.path import basename,exists,join,splitext
from pkg_resources import resource_filename

# text detection model path
MODEL_PATH = resource_filename('campvideo','models/frozen_east_text_detection.pb')
# compatible image types
IMAGE_EXTS = ('.bmp','.jpg','.jpeg','.png','.tiff','.tif')

# load neural net for text bounding box detection
if exists(MODEL_PATH):
    text_net = cv2.dnn.readNet(MODEL_PATH)
else:
    raise Exception('Model data not installed. Please install the models by '
                    'typing the following command into the command line:\n\n'
                    'download_models')

# image class for text and face recognition on video keyframes
class Keyframes(object):
    def __init__(self,ims,names=None):
        """Video keyframes class for image text detection and face recognition
        
        Args:
            ims : array_like
                The list or numpy array of BGR images representing the 
                keyframes of a video
            names : array_like, optional
                An array of the corresponding names for each keyframe in `ims`.
                Generally, these names usually refer to the index of the frame
                in the original video.
        """
        # list of BGR images
        self.ims = list(ims)
        # extension for image type, always set to .png'
        self.ext = '.png'
        # image array properties
        if names is not None: self.names = names
        self.resolution = ims[0].shape[:2]

    # construct from directory of images
    @classmethod
    def fromdir(cls,im_path):
        """Construct a Keyframes object from a directory of images
        
        Args:
            im_path : string
                The path to the directory containing the keyframes, saved as 
                images. The `fromdir` method will attempt to read in all files
                saved in .bmp, .jpg, .jpeg, .png, .tiff, .tif format.
        """
        ims = []
        names = []
        for fname in os.listdir(im_path):
            if fname.endswith(IMAGE_EXTS):
                # read image and get filetype
                ims.append(imread(join(im_path,fname)))
                names.append(splitext(fname)[0])
                
        # check if ims is empty
        if len(ims) == 0:
            warnings.warn("No images found in directory `%s`" % im_path)
        
        return cls(ims,names=names)
    
    # construct from video and keyframe indices
    @classmethod
    def fromvid(cls,vid_path,kf_ind):
        """Construct a Keyframes object from a video file and an array of 
        indices
        
        Args:
            vid_path : string
                The path to the video file.
            kf_ind : array_like
                An array of keyframe indices.
        """
        vid = Video(vid_path)
        ims = vid.frames(frame_inds=kf_ind)
        fname = splitext(basename(vid_path))[0]
        names = ["%s_%04d" % (fname,ind) for ind in kf_ind]
        
        return cls(ims,names=names)

    # show keyframes
    def show(self,wait=None):
        """ Displays the keyframes
        
        Args:
            wait : int, optional
                Delay between showing consecutive images, in milliseconds.
                Default value is infinite, instead waiting for a key press from
                the user to display the next keyframe.
        """
        if wait is None: wait = 0
        
        for i,im in enumerate(self.ims):
            cv2.imshow("Keyframe %d" % (i+1),im)
            cv2.waitKey(wait)
            cv2.destroyAllWindows()

    # function for converting specified image to a byte string
    def tobytes(self,im):
        """ Convert a given image to a byte-encoded string
        
        Args:
            im : numpy array
                BGR numpy array representing the image to be byte-encoded.
        """
        flag,enc = cv2.imencode(self.ext,im)
        # catch errors with encoding
        if flag:
            return enc.tobytes()
        else:
            raise Exception('Unable to encode image')

    # function for detecting and recognizing image text using Google Cloud API
    def image_text(self,bb_thr=0.035,bb_count=25,plot_images=False):
        """ Detect and recognize artificial or scene text in the image

        Args:
            bb_thr : float, optional
                Minimum relative height of detected bounded for text to be
                kept. Default value is 0.035 (3.5% of image height).
            bb_count : int, optional
                Maximum number of words to return. Keeps the `count` largest
                (by height) bounding boxes. Default value is 25.
            plot_images : bool, optional
                Flag for plotting images with detected text bounding boxes
                overlaid. Default value is False.
        """
        # checks of Google API has already been called for keyframes
        if not hasattr(self,'_texts'):
            # flag for detected text
            keep = [True if self._has_text(im) else False for im in self.ims]
            self._keep = keep
            
            # convert to byte strings and check if any image has text
            contents = [self.tobytes(im) for i,im in enumerate(self.ims)
                        if keep[i]]
            if len(contents) == 0: 
                self._texts = []
                return []
            
            # build array of image annotation requests in batches of size 5
            n_batches = int(ceil(len(contents) / 5))
            features = [vision.types.Feature(type='TEXT_DETECTION')]
            requests = [[vision.types.AnnotateImageRequest(
                                image=vision.types.Image(content=content),
                                features=features)
                        for content in contents[5*i:5*i+5]]
                            for i in range(n_batches)]

            # google cloud vision client
            client = vision.ImageAnnotatorClient()
            
            # responses
            responses = [client.batch_annotate_images(batch,timeout=60).responses
                         for batch in requests]

            # store results for future use
            self._texts = [response.text_annotations for batch in responses
                           for response in batch]

        # plot image with bounding boxes
        if plot_images:
            text_ind = np.where(self._keep)[0]
            # deep copy of images
            text_ims = [im.copy() for flag,im in zip(self._keep,self.ims) if flag]

            # filtered list of text
            res = []
            for frame,all_texts in enumerate(self._texts):
                texts = all_texts[1:] # first entry is all text in the image
                rel_heights = []
                inds = []
                for ind,text in enumerate(texts):
                    vertices = np.array([(vertex.x, vertex.y)
                                         for vertex in text.bounding_poly.vertices],
                                        dtype=np.int32) # polylines requires int32
                    # get rotated bb height (assumes rotation is < 90 degrees)
                    dx = vertices[0][0] - vertices[3][0]
                    dy = vertices[0][1] - vertices[3][1]
                    height = ceil(np.sqrt(dx ** 2 + dy ** 2))
                    rel_height = height / self.resolution[0]
        
                    # filter bb based on relative height and total bb count
                    if rel_height >= bb_thr:
                        # insertion index that sorts based on asecending height
                        ins = bisect(rel_heights,rel_height)
                        # sorted heights and corresponding index in texts
                        rel_heights.insert(ins,rel_height)
                        inds.insert(ins,ind)
                        bb = polylines(text_ims[frame],[vertices],
                                       True,(36,255,12))
                    else:
                        bb = polylines(text_ims[frame],[vertices],
                                       True,(0,0,255))
                # plot keyframe with bounding boxes        
                imshow("keyframe %d with bounding boxes" % text_ind[frame],bb)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
                # keep `count` largest texts, in order as they appear in the image
                res.append([texts[i].description for i in sorted(inds[-bb_count:])])
        # no plotting
        else:
            res = []
            for frame,all_texts in enumerate(self._texts):
                texts = all_texts[1:] # first entry is all text in the image
                rel_heights = []
                inds = []
                for ind,text in enumerate(texts):
                    vertices = np.array([(vertex.x, vertex.y)
                                         for vertex in text.bounding_poly.vertices],
                                        dtype=np.int32) 
                    dx = vertices[0][0] - vertices[3][0]
                    dy = vertices[0][1] - vertices[3][1]
                    height = ceil(np.sqrt(dx ** 2 + dy ** 2))
                    rel_height = height / self.resolution[0]
        
                    if rel_height >= bb_thr:
                        ins = bisect(rel_heights,rel_height)
                        rel_heights.insert(ins,rel_height)
                        inds.insert(ins,ind)
        
                res.append([texts[i].description for i in sorted(inds[-bb_count:])])
            
        return res

    # function for checking if text is contained in the image using the EAST
    # text detector from cv2. The purpose of this function is to avoid making
    # API calls to the GCP when no text is present in the image.
    def _has_text(self,im,score_thr=0.5):
        # resize image to nearest dimensions that are a multiple of 32
        h,w = self.resolution
        # resize dimensions, detection is somewhat sensitive to this
        ar = w / h # aspect ratio
        new_w = min(480,w) // 32 * 32
        # nearest (in aspect ratio) height that is a multiple of 32
        new_h = int(np.round(int(new_w/ar) / 32)) * 32
        # make copy to preserve original image
        copy = im.copy()
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
