import campvideo.audio as audio
import campvideo.image as image
import campvideo.text as text
import cv2
import ffmpeg
import io
import numpy as np
import os
import pandas as pd

from campvideo.image import OBAMA_ENC
from campvideo.text import VOCAB
from functools import partial
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
    
    def __init__(self, fpath):
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
    def frames(self, frame_ind=None, size=None, colorspace='BGR'):
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
            frames = np.empty((n, size[1], size[0]), dtype='uint8')
        else:
            frames = np.empty((n, size[1], size[0], 3), dtype='uint8')

        # open video capture object
        self.__stream = cv2.VideoCapture(self.file)

        # read frames from queue
        for i,cur_ind in enumerate(frame_ind):
            ret = self.__stream.set(1, cur_ind)
            # error handling
            if not ret:
                raise Exception('Error setting stream to frame %d' % (cur_ind+1))
            flag, cf = self.__stream.read()
            if not flag:
                raise Exception('Error decoding frame %d' % (cur_ind+1))
                
            # resize
            if (cf.shape[1], cf.shape[0]) != size:
                cf = cv2.resize(cf, size)
            # convert colorspace
            if colorspace != 'BGR':
                cf = cv2.cvtColor(cf, colors[colorspace])

            frames[i] = cf

        # close stream
        self.__release()

        return frames
    
    # function for showing specified frames
    def show(self, frame_ind, wait=None):
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
        
        for i,im in zip(frame_ind, frames):
            cv2.imshow("Frame %d" % i, im)
            cv2.waitKey(wait)
            cv2.destroyAllWindows()
    
    # function for transcribing audio using GCP
    def transcribe(self, phrases=[], use_punct=False):
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
        
        # feature request
        features = [vi.Feature.SPEECH_TRANSCRIPTION]
        
        # construct SpeechTranscriptionConfig object
        config = vi.SpeechTranscriptionConfig(
                        language_code='en-US',
                        speech_contexts=[
                                vi.SpeechContext(phrases=phrases)
                                ],
                        enable_automatic_punctuation=use_punct
                        )
        
        # construct VideoContext
        context = vi.VideoContext(speech_transcription_config=config)
        
        # construct annotation request and get results
        # uri = 'gs://video_files/' + blob.name
        with io.open(self.file, "rb") as movie:
            input_content = movie.read()
            
        request = vi.AnnotateVideoRequest(input_content=input_content,
                                          features=features,
                                          video_context=context)
        operation = v_client.annotate_video(request)
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
            rfpath = join(temp, 'resized.mp4')
            # resize video to 320 x 240
            cmd = ffmpeg.input(self.file,
                       ).output(rfpath, vf='scale=320:240', loglevel=16
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
    
            labhist_feat = np.zeros((n, labhist.dimension),dtype=np.float32)
            hog_feat = np.zeros((n, hog.dimension),dtype=float)
            
            # weight array for computing intensity
            w_intensity = np.array([0.114, 0.587, 0.299])
    
            stream = cv2.VideoCapture(v.file)
            for i in frame_inds:            
                # read frame
                flag, cf = stream.read()
                # error handling
                if not flag:
                    raise Exception('Error decoding frame %d' % (i + 1))
                    
                # reject monochramatic frames (determined by intensity std)
                if no_mono and (cf @ w_intensity).std() < mono_thresh:
                    labhist_feat[i] = np.nan
                    hog_feat[i] = np.nan
                # compute features
                else:
                    labhist_feat[i] = labhist.compute(cf)
                    hog_feat[i] = hog.compute(cf)
    
            # close stream
            stream.release()

        return (labhist_feat, hog_feat)

    # function for adaptively selecting keyframes via submodular optimization
    def summarize(self, l1=1.5, l2=3.5, niter=25, rng=None):
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
            
        rng : numpy RandomGenerator, optional
            Numpy random number generator. By default, the code uses the 
            current state of the RandomGenerator. To specify your own generator,
            instantiate 
            
            rng = np.random.default_rng(SEED)
            
            where `SEED` is some integer, and pass rng to this method as a 
            keyword.
        """
        # random number generator
        rng = np.random.default_rng() if rng is None else rng                
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
        d = -0.5 * pairwise_kernels(lab, metric='additive_chi2', n_jobs=-1)

        # computes objective function value for a given index set
        def objective(kf_ind):
            n_summ = len(kf_ind)

            # representativeness
            r = w[kf_ind].max(axis=0).sum()
            # uniqueness
            d_sub = d[np.ix_(kf_ind, kf_ind)]
            u = 0
            for j in range(1, len(d_sub)):
                # minumum distance to all previously added frames
                u += d_sub[j,:j].min()

            return r + l1*u + l2*(n-n_summ)

        # keyframe index set
        best_obj = 0

        # submodular optimization
        for _ in range(niter):
            # random permutation of frame indices
            u = rng.permutation(n)

            # initial keyframe index sets
            X = np.empty(0, dtype=int)
            Y = np.arange(n)

            # initial distances (similarity)
            wX_max = np.zeros(n, dtype=float)
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
                df_X =   np.maximum(0, w[uk] - wX_max).sum() + l1*dX_min - l2
                # change in objective function for Y set
                df_Y = -(np.maximum(0, w[uk] - wY_max_t).sum() + l1*dY_min - l2)

                # probabilistically add/remove uk
                a = max(df_X,0.0)
                b = max(df_Y,0.0)
                if a + b == 0.0:
                    prob_X = 0.5
                else:
                    prob_X = a / (a + b)

                # add to X, no change in Y
                if rng.uniform() < prob_X:
                    X = np.insert(X, np.searchsorted(X, uk), uk)
                    # update maximum similarity between each i and current X
                    wX_max = np.maximum(wX_max, w[uk])
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

        # rescale indices due to possible differences between resized video 
        # frame count and original length
        best_ind = (self.frame_count * index_sub[best_kf_ind]) // n_res
        return best_ind

    # function for getting all prediction labels for replicated WMP variables
    def encode(self, face_rec=True, music_mood=True, iss_ment=True, 
               cand_ment=True, sentiment=True, use_imtext=True, fav_path=None,
               opp_path=None, fav_name=None, opp_name=None, transcript=None,
               verbose=False):
        """ Predicts labels for specified WMP variables. By default, all
        variables are predicted.

        Parameters
        ----------
        face_rec : bool, optional
            Flag for whether or not to predict candidate and opponent 
            appearances. If true, the user must specify a file path to a 
            reference image using the `fav_path` and `opp_path` keywords for
            the favored candidate and opposing candidate, respectively.
            
        music_mood : bool, optional
            Flag for whether or not to predict music mood, which classifies 
            the audio in the ad as any of ominous / tense, uplifting, or 
            sad / sorrowful.
            
        iss_ment : bool, optional
            Flag for whether or not to detect issue mentions in the ad.
            
        cand_ment : bool, optional
            Flag for whether or not to detect favored and opposing candidate
            mentions in the ad.
            
        sentiment : bool, optional
            Flag for whether or not to classify the sentiment of the transcript
            for the ad.
            
        use_imtext : bool, optional
            Flag for whether or not to use image text in detecting issue 
            mentions and candidate mentions.
            
        fav_path : str, optional
            File path pointing to the reference image for the favored 
            candidate.
            
        opp_path : str, optional
            File path pointing to the reference image for the opposing
            candidate.
            
        fav_name : str, optional
            Favored candidate's name, as used in the campaign. This variable
            must be specificed if `cand_ment = True` and should be formatted as
            follows: 
                
            <FIRST NAME> <1ST LAST NAME> <2ND LAST NAME> <SUFFIX>'
            
            Some examples:
            
            fav_name = 'Bernie Sanders'
            fav_name = 'Alison Lundergan Grimes'
            fav_name = 'Joyce Healy-Abrams'
            fav_name = 'Joseph Kennedy III'
            
            For the best results, use the name that is typically used in the
            campaign ads. In the case of multiple last names, the function will
            check both hyphenated and unhyphenated versions. 
            
        opp_name : str, optional
            Opposing candidate's name, as used in the campaign. This variable
            must be specificed if `cand_ment = True` and should be formatted as
            follows: 
                
            <FIRST NAME> <1ST LAST NAME> <2ND LAST NAME> <SUFFIX>'
            
            Some examples:
            
            opp_name = 'Bernie Sanders'
            opp_name = 'Alison Lundergan Grimes'
            opp_name = 'Joyce Healy-Abrams'
            opp_name = 'Joseph Kennedy III'
            
            For the best results, use the name that is typically used in the
            campaign ads. In the case of multiple last names, the function will
            check both hyphenated and unhyphenated versions. 
            
        transcript : str, optional
            File path pointing to a text file containing the transcript for 
            the video. If no transcript is provided, a transcript will be
            generated using the Google Cloud Platform videointelligence API.
            
        verbose : bool, optional
            Flag for specifying whether or not encoding progress is displayed
            to the user. By default, this variable is false.
        """ 
        
        # if face recognition enabled, check that reference images were
        # provided
        if face_rec and (fav_path is None or opp_path is None):
            raise Exception("missing candidate reference images, please specify "
                "image paths using the `fav_path` and `opp_path` keywords")
            
        # if candidate mentions enabled, check that candidate names were
        # provided
        if cand_ment and (fav_name is None or opp_name is None):
            raise Exception("missing candidate names, please specify the candidate"
                "names using the `fav_name` and `opp_name` keywords")
            
        # check if user has Google Cloud Platform credentials when needed
        if (transcript is None) and (iss_ment or cand_ment or sentiment) or use_imtext:
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                raise KeyError('no API key found for Google Cloud Platform')              
        
        # compute video summary if face_rec enabled or use_imtext is true
        if face_rec or use_imtext:
            if verbose: print('    Computing video summary...', end=' ', flush=True)
            kf_ind = self.summarize()
            kf = image.Keyframes.fromvid(self.file, kf_ind)
            if verbose: print('done!')
            
        # face recognition if enabled
        if face_rec:
            if verbose: print('    Recognizing candidate faces...', end=' ', flush=True)
            f_picture = kf.facerec(fav_path)
            o_picture = kf.facerec(opp_path)
            if verbose: print('done!')
        else:
            f_picture = None
            o_picture = None
            
        # also check if Obama pictured in ad if we have keyframes
        if face_rec or use_imtext:
            obama_picture = kf.facerec(OBAMA_ENC, dist_thr=0.5156)
        else:
            obama_picture = False
            
        # compute transcript if not provided
        if (transcript is None) and (iss_ment or cand_ment or sentiment):
            if verbose: print('    Transcribing video with GCP...', end=' ', flush=True)
            # get hint phrases from provided names
            phrases = list(filter(None, [fav_name, opp_name]))
            # create Text object
            transcript = text.Text(self.transcribe(phrases=phrases).lower())
            if verbose: print('done!')
            
        # image text recognition if enabled
        if use_imtext:
            if verbose: print('    Recognizing image text with GCP...', end=' ', flush=True)
            imtext_raw = kf.image_text()
            # remove empty lists
            imtext_raw = [item for item in imtext_raw if item != []]
            # create Text object
            imtext = text.Text(' '.join(imtext_raw.lower()
                                 ).replace('.', ' ')      # deals with URLs
                              )
            # create transcript object
            if verbose: print('done!')
            
        # music mood classification
        if music_mood:
            if verbose: print('    Classifying music mood...', end=' ', flush=True)
            aud = audio.Audio(self.file)
            mood = aud.musicmood()
            if verbose: print('done!')
        else:
            mood = pd.DataFrame(None, columns=['music1', 'music2', 'music3'],
                                index=[self.title])
            
        # issue mention
        if iss_ment:
            if verbose: print('    Detecting issue mentions...', end=' ', flush=True)
            # transcript
            iss_trans = transcript.issue_mention(include_names=True, 
                                                 include_phrases=True)
            # image text
            if use_imtext:
                iss_im = imtext.issue_mention(include_names=True, 
                                              include_phrases=True)
                # ignore visual data for congress (`congmt`)  and wall street 
                # (`mention16`)
                iss_im['congmt'] = 0
                iss_im['mention16'] = 0
            else:
                iss_im = False
            # combine image and transcript results
            iment = iss_trans | iss_im
            # obama depctions count as president mention
            iment['prsment'] = iment['prsment'] | obama_picture
            if verbose: print('done!')
        else:
            iment = pd.DataDrame(None, columns=VOCAB.wmp, index=[self.title])
            
        # candidate mention
        if cand_ment:
            if verbose: print('    Detecting candidate mentions...', end=' ', flush=True)
            # transcript
            oment_trans = transcript.opp_mention(opp_name)
            fment_trans = transcript.fav_mention(fav_name)
            # image text
            oment_im = imtext.opp_mention(opp_name) if use_imtext else False
            fment_im = imtext.fav_mention(fav_name) if use_imtext else False
            # combine image and transcript results
            o_mention = oment_trans | oment_im
            f_mention = fment_trans | fment_im
            if verbose: print('done!')
        else:
            o_mention = None
            f_mention = None
            
        # sentiment
        if sentiment:
            if verbose: print('    Detecting transcript sentiment...', end=' ', flush=True)
            tone = transcript.sentiment()
            if verbose: print('done!')
        else:
            tone = None
            
        # store results as a pandas DataFrame
        res1 = pd.DataFrame(data={'f_mention': f_mention, 'o_mention': o_mention,
                                  'f_picture': f_picture, 'o_picture': o_picture,
                                  'tone': tone}, index=[self.title])
        res = pd.concat([res1, mood, iment], axis=1)
        res.index.name = 'uid'
        
        return res
        
class LabHistogram:
    def __init__(self, size, nbins=23):
        self.dimension = nbins ** 3
        self.numel = np.prod(size)
        self.__hist = partial(cv2.calcHist, channels=[0,1,2], mask=None,
                              histSize=[nbins,]*3, ranges=[0,256]*3)

    def compute(self,frame):
        counts = self.__hist([cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)])
        return cv2.normalize(counts, counts, norm_type=cv2.NORM_L1).flatten()

class HOG:
    def __init__(self, size, nbins=9, block_size=(16,16), cell_size=(8,8)):
        self.numel = np.prod(size)
        self.__detector = cv2.HOGDescriptor(_winSize=size,_blockSize=block_size,
                                            _blockStride=(8,8), _cellSize=cell_size,
                                            _nbins=nbins)
        self.dimension = self.__detector.getDescriptorSize()

    def compute(self,frame):
        return self.__detector.compute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).flatten()
