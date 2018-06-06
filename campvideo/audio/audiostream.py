from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

from numpy.fft import fft
from scipy.fftpack import dct
from scipy.io.wavfile import read
from scipy.signal import lfilter
from subprocess import call

# calls ffmpeg to extract .wav file from .mp4 file
def extract_audio(fpath,fs=22050):
    # split extension from filename
    fname,ext = os.path.splitext(fpath)
    
    # ouput filename
    out = fname + '.wav'
    
    # build ffmpeg command
    cmd = ['ffmpeg','-y','-loglevel','quiet','-i',fpath,
           '-vn','-ac',str(1),'-ar',str(fs),out]
    
    # run command
    call(cmd)
    
    return out

# create feature vector using all of the features (with default parameters)
def feat(aud_file):
    # spawn AudioFeats object
    aud = AudioStream(aud_file)
    
    # short term timbre features
    mfcc = aud.mfcc()
    osc = aud.osc()
    sfm_scm = aud.sfm_scm()
    ssd = aud.ssd()
    
    feat_st = np.hstack((ssd,
                         mfcc.mean(0),mfcc.std(0),
                         osc.mean(0),osc.std(0),
                         sfm_scm.mean(0),sfm_scm.std(0)))
                      
    # long-term modulation spectral features
    mmfcc = aud.mfeat_spect(mfcc)
    mosc = aud.mfeat_spect(osc)
    msfm_scm = aud.mfeat_spect(sfm_scm)
    
    feat_lt = np.hstack((mmfcc,mosc,msfm_scm))
    
    # joint frequency feature
    joint_feats = aud.joint_feats()
    
    return np.hstack((feat_st,feat_lt,joint_feats))
    
# creates triangular filter bank on mel scale
def trifil_mel(f_min,f_max,nfft,fs,b=25):
    # number of frequency bins (assumes even length FFT)
    nbins = int(nfft // 2) + 1
    # mel scale filter bank
    fbank = np.empty((b,nbins),dtype=float)
    # 50% overlap filter bank
    cp_f = mel2hz(np.linspace(hz2mel(f_min),hz2mel(f_max),b+2))
    # get bin numbers for critical bands (rounding down)
    cp_bin = (nfft * cp_f // fs).astype(int)
        
    # create mel-scale triangle filterbank
    for i in range(len(cp_bin[:-2])):
        fl = cp_bin[i]
        fc = cp_bin[i+1] 
        fu = cp_bin[i+2]
        # generator for filter gains
        gen = (float(k - fl) / (fc - fl) if k >= fl and k <= fc else 
               float(fu - k) / (fu - fc) if k > fc and k <= fu else 
               0 for k in range(nbins))
               
        fbank[i,:] = np.fromiter(gen,dtype=float)
    
    return fbank
    
# creates octave scale (doubling) bandpass filters
# assumes 0 is starting frequency and fs / 2 is ending frequency
# output is the cutpoints of the bands (see numpy.split)
def bpf_osc(nfft,fs):
    # number of bands - 1
    d = int(np.log2((fs-1) / 200.0)) + 1 # first cutoff freq is 100 Hz
    bpf_freq = 100 * 2 ** np.arange(d)
    # convert to indices
    bpf_bin = np.ceil(nfft * bpf_freq / fs).astype(int) # rounding up because of how indexing in python works (exclusive on upper index)
    
    return bpf_bin

# convert Hz to mel scale
def hz2mel(freq):
    """Converts Hz to mel scale.
    
    Args:
        freq: Frequency in Hz
    """
    return 1127*np.log(1 + freq / 700.0)

# convert mel to Hz scale
def mel2hz(freq):
    """Converts mel to Hz scale.
    
    Args:
        freq: Frequency in mels
    """
    return 700*(np.exp(freq / 1127.0) - 1)
    
# function for computing spectrogram (magnitude spectrum)
def spectrogram(y,fs,win_length=1024,wfunc=np.hamming,nfft=None,overlap=0.5,
                pre_emph=0.95):
    # check that win_length is even (will cause issues otherwise)
    if win_length % 2 != 0:
        raise ValueError('win_length must be an even value')
    # convert to single channel if dual channel audio
    if len(y.shape) >= 2:
        y = y.mean(1)
    # set nfft if not set
    if nfft is None:
        nfft = win_length
        
    # spectrogram properties
    ny = len(y)
    noverlap = int(overlap*win_length)
    stride = win_length - noverlap
    nframes = (ny - noverlap) // stride
    
    # window array
    window = wfunc(win_length)
    
    # frequency and frame times
    f = np.arange(0,nfft//2+1) * (fs / nfft)
    t = (np.arange(nframes)*stride + win_length//2) / fs
    
    # split into overlapping segments
    segs = np.array([y[i*stride:i*stride+win_length] for i in range(nframes)])
                     
    # pre-emphasize, window, and compute FFT
    spectrum = fft(lfilter([1,-pre_emph],1,segs,axis=1)*window,nfft,axis=1)
       
    # compute single-sided magnitude spectrum spectrogram
    spectrum = np.abs(spectrum[:,:nfft//2+1]) / nfft
    spectrum[:,1:-1] *= 2
        
    return f,t,spectrum
                              
# audio stream object
class AudioStream(object):
    """Object for keeping track of audio properties, computing spectrogram, and
    extracting various features such as the MFCC, OSC, and SFM/SCM. Audio file 
    must be in WAV format.
    """    
    # constructor
    def __init__(self,audio_file,normalize=False,win_length=1024,overlap=0.5,
                       pre_emph=0.95,wfunc=np.hamming):
        # NOTE: I'm always assuming an even-lengthed window (and hence FFT)
        # read audio file
        self.fs,y = read(audio_file)
        # convert signal to float and normalize to lie between -1.0 and 1.0
        if normalize:
            self.signal = y.astype(np.float16) / (2**15) # float16
        else:
            self.signal = y # int16
        # FFT length
        self.nfft = win_length
        # overlap (in percentage)
        self.overlap = overlap
        
        # frame rate of spectrogram
        self.fs_mod = self.fs / int(self.nfft*self.overlap) - 1
            
        # compute magnitude spectrum spectrogram
        self.freqs,self.time,self.spectrogram = spectrogram(y,self.fs)
    
    # statistical spectrum descriptor
    def ssd(self):
        # signal attributes
        spect = self.spectrogram
        freqs = self.freqs
        
        # signal statistics
        spect_sum = spect.sum(1,keepdims=True)
        spect_mean = spect_sum / spect.shape[1]
        spect_std = spect.std(1)
        
        # centroid
        cent = np.divide((freqs * spect).sum(1),spect_sum.squeeze(),
                         out=np.zeros(len(spect)),where=spect_sum.squeeze() != 0)
        
        # flux 
        flux = np.linalg.norm(np.diff(spect,axis=0),axis=1) ** 2
        
        # rolloff
        roll = freqs[np.argmax(spect.cumsum(1) >= 0.85*spect_sum,axis=1)]
        
        # skewness
        skew = np.divide(((spect - spect_mean) ** 3).mean(1), 
                         spect_std ** 3,out=np.zeros(len(spect)),
                         where=spect_std != 0)
        
        # kurtosis
        kurt = np.divide(((spect - spect_mean) ** 4).mean(1), 
                         spect_std ** 4,out=np.zeros(len(spect)),
                         where=spect_std != 0)
        
        # concatenate feature means and standard deviations
        feat = np.hstack((cent.mean(),cent.std(),
                          flux.mean(),flux.std(),
                          roll.mean(),roll.std(),
                          skew.mean(),skew.std(),
                          kurt.mean(),kurt.std()))
                          
        return feat
        
    # Mel-frequency cepstral coefficients
    def mfcc(self,d=20,b=25,f_min=300,f_max=None):
        """ Computes the Mel-frequency cepstral coefficients for the audio 
        file.
        
        Args:
            d:     Number of coefficients to keep (default is 20)
            b:     Number of bandpass filters (default is 25)
            f_min: Minimum frequency of bandpass filter bank (default is 300)
            f_max: Maximum frequency of bandpass filter bank (default is Fs/2)
        """
        if f_max is None:
            f_max = self.fs/2
            
        # signal attributes
        spect = self.spectrogram
        fs = self.fs
        nfft = self.nfft
        
        # mel-scale filter bank
        fbank = trifil_mel(300,f_max,nfft,fs)
            
        # power spectral density
        psd = (spect ** 2) * nfft
        psd[:,1:-1] *= 0.5
        
        # compute energy in each band of mel-scale filter bank
        eb = np.inner(psd,fbank)
        
        # take DCT to compute mfcc
        mfcc = 0.5*dct(np.log10(1+eb),type=2,n=d,axis=1)
        # append frame energy to mfcc
        mfcc = np.column_stack((mfcc,psd.sum(1)))
            
        return mfcc
        
    # Octave-scale spectral contrast
    def osc(self):
        """ Computes the octave-based spectral contrast for the audio file.
        
        Args:
        """
        # signal properties
        spect = self.spectrogram
        nfft = self.nfft
        fs = self.fs

        # octave-scale band pass filters        
        bpf_bin = bpf_osc(nfft,fs)
                                              
        # neighborhood factor (20% subband length)
        alpha = 0.2 
        
        # split spectrum into subbands and sort
        spect_sub = [np.sort(sub,axis=1) 
                     for sub in np.split(spect,bpf_bin,axis=1)]
        # neighborhood sizes
        lengths = np.hstack((bpf_bin[0],np.diff(bpf_bin),nfft//2+1-bpf_bin[-1]))
        sizes = np.ceil(alpha*lengths).astype(int)
        
        # compute peaks and valleys for each band in each frame
        valleys = np.array([sub[:,:sizes[i]].mean(axis=1) 
                            for i,sub in enumerate(spect_sub)]).T
        peaks = np.array([sub[:,lengths[i]-sizes[i]:].mean(axis=1) 
                          for i,sub in enumerate(spect_sub)]).T
                          
        # take logarithm (add eps to avoid problems with 0)
        valleys = np.log(valleys+np.finfo(float).eps)
        peaks = np.log(peaks+np.finfo(float).eps)
        
        # store in feature array and return
        osc = np.column_stack((valleys,peaks-valleys))
                
        return osc
        
    # spectral flatness and spectral crest
    def sfm_scm(self):
        """ Computes the octave-based spectral flatness measure and spectral
        crest measure for the audio file.
        
        Args:
        """
        # signal properties
        spect = self.spectrogram
        nfft = self.nfft
        fs = self.fs
        
        # acoustic frequency bandpass filters (octave scale)
        bpf_bin = bpf_osc(nfft,fs)
        
        # split into subbands
        spect_sub = np.split(spect,bpf_bin,axis=1)
        
        # subband statistics
        lengths = np.hstack((bpf_bin[0],np.diff(bpf_bin),nfft//2+1-bpf_bin[-1]))
        sub_means = [sub.mean(1,keepdims=True) for sub in spect_sub]
                              
        # compute sfm and scm in each subband
        sfm = np.hstack([np.divide((sub ** (1/lengths[i])).prod(1,keepdims=True),
                         sub_means[i],
                         out=np.zeros((len(spect),1)),
                         where=sub_means[i] != 0) 
                         for i,sub in enumerate(spect_sub)])
                         
        scm = np.hstack([np.divide(sub.max(1,keepdims=True),
                         sub_means[i],
                         out=np.zeros((len(spect),1)),
                         where=sub_means[i] != 0)
                         for i,sub in enumerate(spect_sub)]) 
                   
        return np.hstack((sfm, scm))
        
    # modulation spectrogram
    def mfeat_spect(self,feat,win_length=256,overlap=0.5,b=7):
        """Function for computing a feature vector of statistical descriptors 
        derived from the modulation feature spectrogram.
        
        Args:
            feat:       n x d feature spectrogram, with d the dimension of the
                        feature and n the number of frames in the input
                        spectrogram
            win_length: The number of input spectrogram frames to be used in the
                        computation of the FFT
            overlap:    Overlap factor between consecutive frames
            b:          The number of modulation subbands for reducing feature
                        dimensionality
        """
        # input spectrogram properties
        n,d = feat.shape
        
        # modulation spectrogram properties
        nfft = win_length
        noverlap = int(overlap*win_length)
        stride = win_length - noverlap
        nframes = (n - noverlap) // stride
        
        # split input spectrogram into overlapping segments (nframes x win_length x d) 
        segs = np.array([feat[i*stride:i*stride+win_length,...] for i in range(nframes)])
        
        # take fft along each feature dimension and truncate to positive frequencies
        mspect = np.abs(fft(segs,n=nfft,axis=1)[:,:nfft//2+1,:]) / nfft
        # map negative frequencies magnitutde to positive frequencies
        mspect[:,1:-1,:] *= 2 
        
        # take average across all segments to obtain modulation spectrogram
        mspect = mspect.mean(0) # win_length (nfft) x d
            
        # split into logarithmically-spaced subbands
        # + 1 to make endpoints inclusive
        mo_bin = (2. ** (np.arange(b-1) - (b-1)) * nfft // 2 + 1).astype(int)       
        mspect_sub = np.split(mspect,mo_bin,axis=0)
        
        # compute peaks and valleys in each subband
        mspect_peak = np.array([sub.max(0) for sub in mspect_sub])
        mspect_vall = np.array([sub.min(0) for sub in mspect_sub])
            
        # compute contrast in each subband
        mspect_cont = mspect_peak - mspect_vall
        
        # form feature vector from statistical descriptors
        mspect_feat = np.hstack((mspect_vall.mean(0),mspect_vall.std(0),
                                 mspect_vall.mean(1),mspect_vall.std(1),
                                 mspect_cont.mean(0),mspect_cont.std(0),
                                 mspect_cont.mean(1),mspect_cont.std(1)))
                                 
        return mspect_feat
        
    # joint frequency modulation spectrogram
    def joint_feats(self,b=7):
        """ Computes the joint frequency modulation spectrogram and the AMSC,
        AMSV, AMSFM, and AMSCM joint frequency features.
        
        Args:
            b: The number of modulation frequency subbands for reducing feature
               dimensionality
        """
        # spectrogram properties
        fs = self.fs
        spect = self.spectrogram
        nframes = spect.shape[0]
        nfft = self.nfft
        
        # joint spectrogram properties
        nfft_mod = int(2 ** np.ceil(np.log2(nframes))) # FFT size (modulation)
        
        # compute joint spectrogram (mod freq x ac freq)
        joint_spect = np.abs(fft(spect,n=nfft_mod,axis=0)[:nfft_mod//2+1,:]) / nfft_mod       
        # scale
        joint_spect[1:-1,...] *= 2
        
        # acoustic frequency bandpass filters (octave scale)
        ac_bin = bpf_osc(nfft,fs)
                              
        # modulation frequency bandpass filters (logarithmic scale)
        mo_bin = (2. ** (np.arange(b-1) - (b-1)) * nfft_mod // 2 + 1).astype(int)
        
        # split joint spectrogram into blocks
        joint_sub = [np.sort(block.flatten()) 
                     for sub_mo in np.split(joint_spect,mo_bin,axis=0)
                     for block in np.split(sub_mo,ac_bin,axis=1)] 
                     
        # number of samples in each subband
        lengths = np.array([len(sub) for sub in joint_sub])
        # neighborhood size
        alpha = 0.2
        sizes = np.ceil(alpha*lengths).astype(int)
        
        # joint frequency peaks and valleys
        amsp = np.array([np.log(sub[lengths[i]-sizes[i]:].mean() + np.finfo(float).eps)
                         for i,sub in enumerate(joint_sub)])
        amsv = np.array([np.log(sub[:sizes[i]].mean() + np.finfo(float).eps)
                         for i,sub in enumerate(joint_sub)])
                         
        # joint frequency flatness and crest measures
        sub_prods = np.array([(sub ** (1 / lengths[i])).prod() 
                              for i,sub in enumerate(joint_sub)]) 
        sub_maxes = np.array([sub[-1] for sub in joint_sub])
        sub_means = np.array([sub.mean() for sub in joint_sub])
                         
        amsfm = np.divide(sub_prods,sub_means,where=sub_means != 0)
        amscm = np.divide(sub_maxes,sub_means,where=sub_means != 0)
        
        # return feature vector
        return np.concatenate((amsv,amsp-amsv,amsfm,amscm))

                          
# main script for generating feature vector of given file or directory of files
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('aud_dir', help='Path to directory containing'
                        ' .mp4 files for feature extraction')
    args = parser.parse_args()
    aud_dir = args.aud_dir
    
    # search for .mp4 files
    fpaths = [os.path.join(root,fn) for root,folder,fnames in os.walk(aud_dir)
                                    for fn in fnames
                                    if fn.endswith('.mp4')]
                                    
    # sort alphabetically by filename (ignoring case)
    fpaths = sorted(fpaths,key=lambda s: os.path.basename(s.lower()))
    
    # video IDS
    ids = [os.path.splitext(os.path.basename(fpath))[0]
           for fpath in fpaths]
                                    
                                
    # text file to keep track of filenames
    file_ids = os.path.join(aud_dir,'video_ids.txt')
    with open(file_ids,'w+') as tf:
        for fid in ids:
            tf.write(fid+'\n')
                    
    # check if no videos found
    if len(fpaths) == 0:
        raise ValueError("no .mp4 files found in specified"
                           " directory '{0}'".format(aud_dir))
                    
    # extract features
    d = 636 # default feature vector length
    feats = np.empty((len(fpaths),d),dtype=float)
    
    # fill feature matrix
    for k,fpath in enumerate(fpaths):
        print('Processing video {0} of {1}...'.format(k+1,len(fpaths)))
        # get .wav file
        cur_file = extract_audio(fpath)
        # compute feature
        feats[k,...] = feat(cur_file)
        # delete .wav file
        os.remove(cur_file)
        
    # save data
    np.save(aud_dir,feats)
