import cv2
import ffmpeg
import io
import numpy as np
import os

from functools import partial
# from google.cloud import storage
from google.cloud import videointelligence as vi
from os.path import join
from sklearn.metrics import pairwise_kernels
from sklearn.metrics.pairwise import cosine_similarity
from tempfile import TemporaryDirectory

class Video:
    """Video class for extracting specific frames, computing the video summary,
    and using the GCP videointelligence API to transcribe the video.
    
    Parameters
    ----------
    fpath : str
        The path to the video file.
    
    Attributes
    ----------
    file : str
        The full path for the video file.
        
    title : str
        The name of the file.
        
    frame_count : int
        The number of frames in the video
        
    fps : float
        The FPS of the video.
        
    duration : float
        The length of the video in seconds.
        
    resolution : tuple
        The resolution of the video (width, height)
        
    transcript : str
        The transcript of the video. Initially unset until `transcribe()` is
        called.
    """
    
    def __init__(self,fpath):
        # check that file exists
        if not os.path.exists(fpath):
            raise IOError("file `%s` does not exist" % fpath)

        # create VideoCapture object
        self.__stream = cv2.VideoCapture(fpath)

        # video properties
        self._file = os.path.abspath(fpath)
        self._title = os.path.splitext(os.path.basename(fpath))[0]
        self._frame_count = int(self.__stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.__stream.get(cv2.CAP_PROP_FPS)
        self._duration = self.frame_count / self.fps
        self._resolution = (int(self.__stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self.__stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # initialize transcript as None
        self._transcript = None

        # close stream
        self.__release()
        
    # attribute reader functions
    @property
    def file(self):
        return self._file
    @property
    def title(self):
        return self._title
    @property
    def frame_count(self):
        return self._frame_count
    @property
    def fps(self):
        return self._fps
    @property
    def duration(self):
        return self._duration
    @property
    def resolution(self):
        return self._resolution
    @property
    def transcript(self):
        return self._transcript

    # close VideoCapture stream
    def __release(self):
        self.__stream.release()

    # method for grabbing specified frames
    def frames(self,frame_ind=None,size=None,colorspace='BGR'):
        """Returns the requested video frames.

        Parameters
        ----------
        frame_ind : array_like, optional
            The indices of the frames to be retrieved. Returns all frames by
            default.
            
        size : tuple, optional
            Two-element tuple specifying the desired size of the frames
            with width as the first element and height as the second
            element. The default is the native resolution.
            
        colorspace : str, optional
            String specifying the colorspace representation of the frames.
            Valid options include 'RGB','BGR','YUV','HSV','Lab', and
            'gray'. The default is 'BGR'.
            
        Returns
        -------
        frames : array_like
            The array containing the extracted frames. The first dimension 
            indexes the frames, the second dimension indexes the the rows of 
            each frame, the third dimension idexes the columns, and the fourth
            dimension indexes the color layers.
        """
        # color conversion dictionary
        colors = {'RGB' : cv2.COLOR_BGR2RGB,'YUV' : cv2.COLOR_BGR2YUV,
                  'HSV' : cv2.COLOR_BGR2HSV,'Lab' : cv2.COLOR_BGR2Lab,
                  'gray' : cv2.COLOR_BGR2GRAY}

        # create iterable if one index specified
        if frame_ind is not None:
            frame_ind = np.atleast_1d(frame_ind)
        if frame_ind is not None and len(frame_ind) == 0:
            return []
        # all frames returned if frame_inds is None
        if frame_ind is None:
            frame_ind = np.arange(self.frame_count)
            
        # frame resolution
        if size is None:
            size = self.resolution

        # check if frame_ind contains out-of-bounds entries
        if frame_ind is not None:
            if np.max(frame_ind) >= self.frame_count or np.min(frame_ind) < 0:
                raise IndexError("frame indices out of bounds. Please specify "
                                 "indices from 0 to %d" % (self.frame_count-1))
            # check that specified colorspace is correct
            if colorspace != 'BGR' and colorspace not in colors:
                raise ValueError("`%s` is not a valid colorspace. Please see "
                                  "function details for a list of valid colorspaces"
                                  % colorspace)

        # number of frames and layers
        n = np.size(frame_ind)
        if colorspace == 'gray':
            frames = np.empty((n,size[1],size[0]),dtype='uint8')
        else:
            frames = np.empty((n,size[1],size[0],3),dtype='uint8')

        # open video capture object
        self.__stream = cv2.VideoCapture(self.file)

        # read frames from queue
        for i,cur_ind in enumerate(frame_ind):
            ret = self.__stream.set(1,cur_ind)
            # error handling
            if not ret:
                raise Exception('Error setting stream to frame %d' % (cur_ind+1))
            flag,cf = self.__stream.read()
            if not flag:
                raise Exception('Error decoding frame %d' % (cur_ind+1))
                
            # resize
            if (cf.shape[1],cf.shape[0]) != size:
                cf = cv2.resize(cf,size)
            # convert colorspace
            if colorspace != 'BGR':
                cf = cv2.cvtColor(cf,colors[colorspace])

            frames[i] = cf

        # close stream
        self.__release()

        return frames
    
    # function for showing specified frames
    def show(self,frame_ind,wait=None):
        """ Displays the keyframes
        
        Parameters
        ----------
        frame_ind : array_like
            Indices of frames to display.
            
        wait : int, optional
            Delay between showing consecutive images, in milliseconds.
            Default value is infinite, instead waiting for a key press from
            the user to display the next keyframe.
        """
        # get frames
        frames = self.frames(np.atleast_1d(frame_ind))
        # wait until key press if not specified
        if wait is None: wait = 0
        
        for i,im in zip(frame_ind,frames):
            cv2.imshow("Frame %d" % i, im)
            cv2.waitKey(wait)
            cv2.destroyAllWindows()
    
    # function for transcribing audio using GCP
    def transcribe(self,phrases=[],use_punct=False):
        """Transcribe the audio of the video using the Google Cloud Platform
        
        Parameters
        ----------
        phrases : list, optional
            A list of strings to use as hints for transcribing the audio.
            Generally, this should be kept short and restricted to phrases
            that are not found in the dictionary (e.g. names, locations).
            The default is an empty list [] (no phrases).
             
        use_punct : bool, optional
            Boolean flag specifying whether or not to include punctuation in
            the transcript. The default is False.
            
        Returns
        -------
        transcript : str
            The auto-generated transcript for the video.
            
        Example
        -------
            vid = Video(vid_file)
            phrases = ['Barack Obama', 'Mitt Romney']
            transcript = vid.transcribe(phrases)
        """
        # check if user has GCP access
        if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
            raise KeyError('no API key found for Google Cloud Platform')
            
        # video client
        v_client = vi.VideoIntelligenceServiceClient()
        # s_client = storage.Client('adclass-1286')
        
        # # upload file to bucket
        # bucket = s_client.get_bucket('video_files')
        # blob = bucket.blob(self.title)
        # blob.upload_from_filename(self.file)
        
        # feature request
        features = [vi.enums.Feature.SPEECH_TRANSCRIPTION]
        
        # construct SpeechTranscriptionConfig object
        config = vi.types.SpeechTranscriptionConfig(
                            language_code='en-US',
                            speech_contexts=[
                                    vi.types.SpeechContext(phrases=phrases)
                                    ],
                            enable_automatic_punctuation=use_punct
                            )
        
        # construct VideoContext
        context = vi.types.VideoContext(speech_transcription_config=config)
        
        # construct annotation request and get results
        # uri = 'gs://video_files/' + blob.name
        with io.open(self.file, "rb") as movie:
            input_content = movie.read()
            
        operation = v_client.annotate_video(input_content=input_content,
                                            features=features,
                                            video_context=context)
        # timeout after 3 mins
        result = operation.result(timeout=180)
        
        # transcript
        results = result.annotation_results[0]
        transcript = ''
        for speech_transcription in results.speech_transcriptions:
            transcript += speech_transcription.alternatives[0].transcript
        
        self._transcript = transcript
        return self._transcript
    
    # method for computing frame features
    def videofeats(self, no_mono=True, mono_thresh=5):
        """Computes the Lab histogram and HOG features for the frames in the
           video. These features are computed from a resized version of the 
           video (320 x 240 resolution).

        Parameters
        ----------
        no_mono : bool, optional
            Boolean flag for specifying whether or not monochramatic frames are
            ignored. If True, the corresponding entry in labhist_feat and 
            hog_feat is an array of numpy.nan. The default is True.
            
        mono_thresh : float, optional
            Threshold for declaring a frame as monochromatic. Frames with 
            intensity standard deviation below this value are declared as 
            monochromatic. The default is 5.
            
        Returns
        -------
        labhist_feat : array_like
            The vectorized Lab color histogram for each frame.
            
        hog_feat : array_like
            The histrogram of oriented gradients (HOG) feature for each frame.
        """     
        # resize video to something more manageable to prevent users from 
        # accidentally overloading resources
        with TemporaryDirectory() as temp:
            # resized file name
            rfpath = join(temp,'resized.mp4')
            # resize video to 320 x 240
            cmd = ffmpeg.input(self.file,
                       ).output(rfpath,vf='scale=320:240',loglevel=16
                       ).overwrite_output()
            try:
                cmd.run(capture_stderr=True)
            except ffmpeg.Error as e:
                raise Exception("resizing video failed with error %s:" %
                                e.stderr.decode('utf-8'))
                
            # instantiate video object
            v = Video(rfpath)

            # frame indices
            frame_inds = np.arange(v.frame_count)
            n = len(frame_inds)
    
            # feature class instantiation
            hog = HOG(v.resolution)
            labhist = LabHistogram(v.resolution)
    
            labhist_feat = np.zeros((n,labhist.dimension),dtype=np.float32)
            hog_feat = np.zeros((n,hog.dimension),dtype=float)
            
            # weight array for computing intensity
            w_intensity = np.array([0.114,0.587,0.299])
    
            v.__stream = cv2.VideoCapture(v.file)
            for i,frame_ind in enumerate(frame_inds):            
                # read frame
                flag,cf = v.__stream.read()
                # error handling
                if not flag:
                    raise Exception('Error decoding frame %d' % (frame_ind+1))
                    
                # reject monochramatic frames (determined by intensity std)
                if no_mono and (cf @ w_intensity).std() < mono_thresh:
                    labhist_feat[i] = None
                    hog_feat[i] = None
                # compute features
                else:
                    labhist_feat[i] = labhist.compute(cf)
                    hog_feat[i] = hog.compute(cf)
    
            # close stream
            v.__release()

        return (labhist_feat, hog_feat)

    # function for adaptively selecting keyframes via submodular optimization
    def summarize(self,l1=1.5,l2=3.5,niter=25):
        """Generates an array of keyframes using an adaptive keyframe selection
           algorithm.

        Parameters
        ----------
        l1 : float, optional
            Penalty for uniqueness. The default is 1.5.
            
        l2 : float, optional
            Penalty for summary length. The default is 3.5.
            
        niter : int, optional
            Number of iterations for optimization algorithm. The default is 25.
        """                
        # get histogram and HOG features
        lab, hog = self.videofeats()
        
        # video length before subsetting
        n_res = len(lab)
        
        # subset down to non-monochromatic frames
        index_sub = np.where(~np.isnan(lab).any(axis=1))[0]
        lab = lab[index_sub]
        hog = hog[index_sub]

        # video length after subsetting
        n = hog.shape[0]

        # pairwise comparisons of features
        w = cosine_similarity(hog)
        d = -0.5 * pairwise_kernels(lab,metric='additive_chi2',n_jobs=-1)

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
                if change_ind.any():
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

        # rescale indices due to possible differences between resized frame
        # count and original length
        return (self.frame_count * index_sub[best_kf_ind]) // n_res

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
