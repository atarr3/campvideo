from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ffmpeg
import numpy as np

from os.path import basename,join,splitext
from scipy.fftpack import dct
from scipy.io.wavfile import read
from scipy.signal import lfilter, spectrogram
from tempfile import TemporaryDirectory

# audio class for computing spectrogram and spectral feats. 
class Audio(object):
    def __init__(self,path,normalize=True,fs=5000,nfft=1024,wfunc=np.hamming,
                  wlen=None,overlap=0.5,pre_emph=0.,scaling='density',mode='psd'):
        # set parameters
        self.path = path
        self.fs,self.nfft = fs,nfft
        self.scaling,self.mode = scaling,mode
        
        # window length defaults to nfft if not specified
        wlen = nfft if wlen is None else wlen
        
        if wlen > nfft: Warning('fft length smaller than window')

        # number of samples overlapping between frames
        noverlap = int(overlap*wlen) if overlap < 1 else overlap

        # get .wav file from fpath
        with TemporaryDirectory() as temp:
            # no need to convert
            if path.endswith('.wav'):
                aud_path = path
            else:
                aud_path = vid2wav(path,out_dir=temp,fs=fs)
            # read in
            _,y = read(aud_path)

        # normalize to -1,1 (16-bit float)
        if normalize: y = y.astype(np.float16) / (2**15)

        # apply pre-emphasis filter to entire signal
        if pre_emph > 0: y = lfilter([1,-pre_emph],1,y)

        # compute window function
        window = wfunc(nfft) if wlen is None else wfunc(wlen)
        
        # scale factor for converting between density and spectrum scaling
        self._dens2spect_scale = fs * (window * window).sum() / window.sum() ** 2

        # compute one-sided spectrogram
        f,t,S = spectrogram(y,fs=fs,window=window,noverlap=noverlap,
                            nfft=nfft,return_onesided=True,scaling=scaling,
                            mode=mode)
        
        # double frequencies for stft mode (not sure why it's not done already)
        if mode != 'psd':
            if self.nfft % 2:
                S[1:,] *= 2
            else:
                S[1:-1,] *= 2

        # set attributes, spectrogram is time x freq
        self.freq,self.time,self.spectrogram = f,t,S.T
        
    # function for converting from psd to magnitude spectrum
    def psd2mag(self,scaling='spectrum'):
        # deep copy of spectrogram (currently the PSD)
        spect = self.spectrogram.copy()
        
        # one-sided rescaling
        if self.nfft % 2:
            spect[...,1:] *= 2
        else:
            spect[...,1:-1] *= 2
            
        # change scaling between density and spectrum
        if scaling != self.scaling:
            if self.scaling == 'density':
                spect *= self._dens2spect_scale  
            else:
                spect /= self._dens2spect_scale
                
        # take square root
        return np.sqrt(spect)
        
    # function for converting from magnitude spectrum to psd
    def mag2psd(self,scaling='density'):
        # deep copy of spectrogram (currently the magnitude spectrum)
        spect = self.spectrogram.copy()
        
        # one-sided rescaling
        if self.nfft % 2:
            spect[...,1:] /= np.sqrt(2)
        else:
            spect[...,1:-1] /= np.sqrt(2)
            
        # swap scaling if specified
        if scaling != self.scaling:
            if self.scaling == 'density':
                spect *= np.sqrt(self._dens2spect_scale)  
            else:
                spect /= np.sqrt(self._dens2spect_scale)
                
        # take square
        return spect ** 2
        

    # function for computing energy in logarithmically spaced bands for each frame
    # in a spectrogram
    def energy(self,nbands=33,fmin=300,fmax=2000):
        # initialization
        fs,nfft = self.fs,self.nfft
        # need psd to compute energy
        S = self.spectrogram if self.mode == 'psd' else self.mag2psd()

        # assert fmax < fs/2
        if fmax >= fs/2:
            raise ValueError('fmax must be below the Nyquist frequency')

        # filterbank critical points
        cp_k = np.round(np.geomspace(fmin * nfft/fs,fmax * nfft/fs,nbands+1)
                       ).astype(int)

        # compute sums in each band and concatenate (this is incorrect when mode != psd)
        eb = np.column_stack([band.sum(axis=1)
                              for band in np.split(S,cp_k,axis=1)[1:-1]
                             ])

        return eb

    # function for computing audio fingerprint from spectrogram
    def fingerprint(self,reliability=True):
        # compute energy in each frame
        eb = self.energy()

        # compute difference for adjacent frames and bands
        eb_diff = np.diff(-np.diff(eb,axis=1),axis=0)

        # binarize according to sign and convert to int and compute reliability
        code = np.fromiter((_bin2int(bits) for bits in eb_diff > 0),dtype=np.uint32)

        # return fingerprint and array index (not bit!) of 3 least reliable bits
        rel = 31-np.argsort(np.abs(eb_diff),axis=1)[...,:3] if reliability else None

        return code,rel
        
    # create feature vector from spectrogram for use in mood classification and
    # sentiment analysis
    def audiofeat(self,feature_set='best'):  
        # check proper feature_set specification
        if feature_set not in {'best','all','no-joint'}:
            raise ValueError("Invalid value for feature_set")
            
        # short term timbre features
        ssd = self._ssd()
        mfcc = self._mfcc()
        osc,sfm_scm = self._osfeats()
        
        feat_st = np.hstack((ssd,
                             mfcc.mean(0),mfcc.std(0),
                             osc.mean(0),osc.std(0),
                             sfm_scm.mean(0),sfm_scm.std(0)
                           ))
                        
        # long-term modulation spectral features
        if feature_set == ('all' or 'no-joint'):
            feat_lt = np.hstack((self._mfeat_spect(mfcc),
                                 self._mfeat_spect(osc),
                                 self._mfeat_spect(sfm_scm)
                               ))
        elif feature_set == 'best':
            feat_lt = self._mfeat_spect(mfcc)
                            
        # joint frequency feature
        if feature_set != 'no-joint':
            joint_feats = self._joint_feats()
        
            return np.hstack((feat_st,feat_lt,joint_feats))
        else:
            return np.hstack((feat_st,feat_lt))
        
    # statistical spectrum descriptor for spectrogram (STFT)
    def _ssd(self):
        """ Computes statistcal measures for the magnitude spectrum, including
        the spectral centroid, spectral flux, spectral rolloff, skewness, and
        kurtosis.
        
        Args:
        """
        # get spectrogram and convert from psd if necessary
        spect = self.spectrogram if self.mode != 'psd' else self.psd2mag()
        # frequencies
        freq = self.freq
        
        # signal statistics
        spect_sum = spect.sum(axis=1,keepdims=True) # sum of frequencies in each frame
        spect_mean = spect_sum / spect.shape[1] # avg magnitude
        spect_std = spect.std(axis=1)
        
        # centroid
        cent = np.divide((freq * spect).sum(axis=1),spect_sum.squeeze(),
                         out=np.zeros(len(spect)),where=spect_sum.squeeze() != 0)
        
        # flux 
        flux = np.linalg.norm(np.diff(spect,axis=0),axis=1) ** 2
        
        # rolloff
        roll = freq[np.argmax(spect.cumsum(axis=1) >= 0.85*spect_sum,axis=1)]
        
        # skewness
        skew = np.divide(((spect - spect_mean) ** 3).mean(axis=1), 
                         spect_std ** 3,out=np.zeros(len(spect)),
                         where=spect_std != 0)
        
        # kurtosis
        kurt = np.divide(((spect - spect_mean) ** 4).mean(axis=1), 
                         spect_std ** 4,out=np.zeros(len(spect)),
                         where=spect_std != 0)
        
        # concatenate feature means and standard deviations
        feat = np.hstack((cent.mean(),cent.std(),
                          flux.mean(),flux.std(),
                          roll.mean(),roll.std(),
                          skew.mean(),skew.std(),
                          kurt.mean(),kurt.std())
                        )
                          
        return feat
        
    # Mel-frequency cepstral coefficients
    def _mfcc(self,d=20,b=25,f_min=300,f_max=None):
        """ Computes the Mel-frequency cepstral coefficients for the audio 
        file.
        
        Args:
            d:     Number of coefficients to keep (default is 20)
            b:     Number of bandpass filters (default is 25)
            f_min: Minimum frequency of bandpass filter bank (default is 300)
            f_max: Maximum frequency of bandpass filter bank (default is Fs/2)
        """
        if f_max is None:
            f_max = self.freq[-1]
            
        # signal attributes
        psd = self.mag2psd() if self.mode != 'psd' else self.spectrogram
        fs = self.fs
        nfft = self.nfft
        
        # mel-scale triangular filter bank
        fbank = _trifil_mel(300,f_max,nfft,fs)
        
        # compute energy in each band of mel-scale filter bank
        eb = np.inner(psd,fbank)
        
        # take DCT to compute mfcc
        mfcc = 0.5*dct(np.log10(1+eb),type=2,n=d,axis=1)
        # append frame energy to mfcc
        mfcc = np.column_stack((mfcc,psd.sum(1)))
            
        return mfcc
        
    # Octave-scale features (spectral contrast, flatness, and crest)
    def _osfeats(self):
        """ Computes the octave-based spectral contrast, spectral flatness
        measure, and spectral crest measure for the spectrogram.
        
        Args:
        """
        # signal properties
        spect = self.spectrogram if self.mode != 'psd' else self.psd2mag()
        nfft = self.nfft
        fs = self.fs

        # octave-scale band pass filter critical points        
        bpf_bin = _bpf_os(nfft,fs,freq='acoustic')
                                              
        # neighborhood factor (20% subband length)
        alpha = 0.2 
        
        # split spectrum into subbands and sort
        spect_sub = [np.sort(sub,axis=1) 
                     for sub in np.split(spect,bpf_bin,axis=1)]
        # subband lengths
        lengths = np.hstack((bpf_bin[0],np.diff(bpf_bin),nfft//2+1-bpf_bin[-1]))
        # neighborhood sizes
        sizes = np.ceil(alpha*lengths).astype(int)
        # subband means
        sub_means = [sub.mean(axis=1) for sub in spect_sub]
        
        # compute peaks and valleys for each band in each frame
        valls = np.array([sub[:,:sizes[i]].mean(axis=1) 
                          for i,sub in enumerate(spect_sub)]).T
        peaks = np.array([sub[:,lengths[i]-sizes[i]:].mean(axis=1) 
                          for i,sub in enumerate(spect_sub)]).T
        
        # take logarithm (add eps to avoid problems with 0)
        valls = np.log(valls + np.finfo(float).eps)
        peaks = np.log(peaks + np.finfo(float).eps)
        # concatenate
        osc = np.column_stack((valls,peaks-valls))
                              
        # compute sfm and scm in each subband
        sfm = np.column_stack([np.divide(
                                   np.exp(np.log(sub,
                                                 out=-np.inf*np.ones(sub.shape),
                                                 where=sub != 0
                                                ).mean(axis=1)),
                                   sub_means[i],
                                   out=np.zeros((len(spect))),
                                   where=sub_means[i] != 0) 
                               for i,sub in enumerate(spect_sub)])
                         
        scm = np.column_stack([np.divide(sub[:,-1],
                               sub_means[i],
                               out=np.zeros((len(spect))),
                               where=sub_means[i] != 0)
                               for i,sub in enumerate(spect_sub)]) 
                
        return osc,np.hstack((sfm, scm))

    # modulation spectrogram
    def _mfeat_spect(self,feat,wlen=256,overlap=0.5):
        """Function for computing a feature vector of statistical descriptors 
        derived from the modulation feature spectrogram.
        
        Args:
            feat:    n x d feature spectrogram, with d the dimension of the
                     feature and n the number of frames in the input
                     spectrogram
            wlen:    The the length of the texture window for computation of the
                     FFT
            overlap: Overlap factor between consecutive frames
        """
        # input spectrogram properties
        n,d = feat.shape
        fs_mo = 1 / (self.time[1]-self.time[0])
        
        # modulation spectrogram properties
        nfft_mo = wlen
        noverlap = int(overlap*wlen)
        
        # compute one-sided magnitude spectrogram along each feature dimension
        # S has dimensions nfft_mo/2 + 1 x d x nframes
        _,_,S = spectrogram(feat,window=np.ones(wlen),noverlap=noverlap,
                            scaling='spectrum',mode='magnitude',axis=0)
                            
        # map negative frequency magnitudes to positive frequencies
        if nfft_mo % 2:
            S[1:,...] *= 2
        else:
            S[1:-1,...] *= 2
        
        # take average across all frames to obtain modulation spectrogram
        mspect = S.mean(axis=-1).T # d x nfft_mo/2 + 1
            
        # split into logarithmically-spaced subbands
        mo_cp = _bpf_os(nfft_mo,fs_mo,freq='modulation') 
        mspect_sub = np.split(mspect,mo_cp,axis=1)
        
        # compute peaks and valleys in each subband
        mspect_peak = np.array([sub.max(axis=1) for sub in mspect_sub])
        mspect_vall = np.array([sub.min(axis=1) for sub in mspect_sub])
            
        # compute contrast in each subband
        mspect_cont = mspect_peak - mspect_vall
        
        # form feature vector from statistical descriptors
        mspect_feat = np.hstack((mspect_vall.mean(0),mspect_vall.std(0),
                                 mspect_vall.mean(1),mspect_vall.std(1),
                                 mspect_cont.mean(0),mspect_cont.std(0),
                                 mspect_cont.mean(1),mspect_cont.std(1)))
                                 
        return mspect_feat
        
    # joint frequency modulation spectrogram
    def _joint_feats(self):
        """ Computes the joint frequency modulation spectrogram and the AMSC,
        AMSV, AMSFM, and AMSCM joint frequency features.
        
        Args:
            b: The number of modulation frequency subbands for reducing feature
               dimensionality
        """
        # spectrogram properties
        fs_ac = self.fs
        fs_mo = 1 / (self.time[1]-self.time[0])
        spect = self.spectrogram if self.mode != 'psd' else self.psd2mag()
        nfft_ac = self.nfft
        nfft_mo = len(spect)
        
        # compute joint spectrogram (mod freq x ac freq)
        _,_,S = spectrogram(spect,window=np.ones(nfft_mo),scaling='spectrum',
                            mode='magnitude',axis=0)
        S = S.squeeze()  
        # map negative frequency magnitudes to positive frequencies
        if nfft_mo % 2:
            S[1:,...] *= 2
        else:
            S[1:-1,...] *= 2
        
        # octave scale critical points (acoustic frequency)
        ac_cp = mo_cp = _bpf_os(nfft_ac,fs_ac,freq='acoustic')
        # octave scale critical points (modulation frequency)
        mo_cp = _bpf_os(nfft_mo,fs_mo,freq='modulation')
        
        # split joint spectrogram into blocks
        joint_sub = [np.sort(sub_ac.flatten()) 
                     for sub_mo in np.split(S,mo_cp,axis=0)
                     for sub_ac in np.split(sub_mo,ac_cp,axis=1)] 
                     
        # number of samples in each subband
        lengths = np.fromiter((len(sub) for sub in joint_sub),dtype=int)
        # neighborhood size
        alpha = 0.2
        sizes = np.ceil(alpha*lengths).astype(int)
        
        # joint frequency peaks and valleys
        amsv = np.array([sub[:sizes[i]].mean()
                         for i,sub in enumerate(joint_sub)])
        amsp = np.array([sub[lengths[i]-sizes[i]:].mean()
                         for i,sub in enumerate(joint_sub)])
                         
        amsv = np.log(amsv + np.finfo(float).eps)
        amsp = np.log(amsp + np.finfo(float).eps)
                         
        # joint frequency flatness and crest measures
        sub_prods = np.array([np.exp(np.log(sub,
                                            out=-np.inf*np.ones(sub.shape),
                                            where=sub != 0).mean()) 
                              for i,sub in enumerate(joint_sub)]) 
        sub_maxes = np.array([sub[-1] for sub in joint_sub])
        sub_means = np.array([sub.mean() for sub in joint_sub])
                         
        amsfm = np.divide(sub_prods,sub_means,out=np.zeros(len(sub_means)),
                          where=sub_means != 0)
        amscm = np.divide(sub_maxes,sub_means,out=np.zeros(len(sub_means)),
                          where=sub_means != 0)
        
        # return feature vector
        return np.concatenate((amsv,amsp-amsv,amsfm,amscm))
        
# function for getting the duration of a media file
def get_dur(med_file):
    fname,ext = splitext(med_file)
    try:
        dur = float(ffmpeg.probe(med_file)['format']['duration'])
        return dur
    except KeyError:
        # no duration key in ffprobe call, caused by issue with the file
        with TemporaryDirectory() as temp:
            fixed = join(temp,'fixed'+ext)
            cmd = ffmpeg.input(med_file
                       ).output(fixed,c='copy',loglevel=16)
            cmd.run()
            return float(ffmpeg.probe(fixed)['format']['duration'])
    except ffmpeg.Error as e:
        raise Exception(e.stderr.decode('utf-8'))

# function for trimming media file to specified start point and duration
def trim(med_file,start,dur):
    # split extension from file
    fname,ext = splitext(med_file)
    # output file
    out_file = fname + '_trimmed' + ext
    # build ffmpeg command
    cmd = ffmpeg.input(med_file,ss=start,t=dur,vn=None
               ).output(out_file,to=dur,loglevel=16
               ).overwrite_output()                
    # call command
    try:
        cmd.run(capture_stderr=True)
        return out_file
    except ffmpeg.Error as e:
        raise Exception(e.stderr.decode('utf-8'))

# function for converting a media file to a .wav file
def vid2wav(vid_file,out_dir=None,fs=5000,channels=1,start=None,dur=None):
    # split extension from file
    fname,ext = splitext(vid_file)
    if out_dir is None:
        out_path = fname + '_proc.wav'
    else:
        out_path = join(out_dir,basename(fname) + '_proc.wav')

    # build ffmpeg command
    if start is None or dur is None:
        cmd = ffmpeg.input(vid_file,vn=None
                   ).output(out_path,ar=fs,ac=channels,loglevel=16
                   ).overwrite_output()
                                     
    else:
        # can't do seeking before reading in stream for .wmv files
        if ext == '.wmv':
            cmd = ffmpeg.input(vid_file,vn=None
                       ).output(out_path,ss=start,t=dur,ar=fs,ac=channels,loglevel=16
                       ).overwrite_output() 
        else:
            cmd = ffmpeg.input(vid_file,ss=start,t=dur,vn=None
                       ).output(out_path,ar=fs,ac=channels,loglevel=16
                       ).overwrite_output() 
    # call command
    try:
        cmd.run(capture_stderr=True)
        return out_path
    except ffmpeg.Error as e:
        raise Exception(e.stderr.decode('utf-8'))

# converts a binary array to an integer
def _bin2int(bit_array):
    res = 0
    for bit in bit_array:
        res = (res << 1) | bit

    return res
    
# computes octave scale (doubling) bandpass filter critical points for a
# specified number of bands. bands specified in original paper (though their
# math is off, and I don't know how to rectify the discrepancy)
def _bpf_os(nfft,fs,freq='acoustic'):
    if freq == 'acoustic':
        # number of bands
        d = 8
        bpf_freq = 100 * 2 ** np.arange(d-1) # first cutoff is 100Hz
    elif freq == 'modulation':
        d = 7
        bpf_freq = 0.33 * 2 ** np.arange(d-1) # first cutoff is 0.33Hz
        
    # convert to indices, rounding up
    bpf_bin = np.ceil(nfft * bpf_freq / fs).astype(int)
    
    return bpf_bin
    
# creates triangular filter bank on mel scale
def _trifil_mel(f_min,f_max,nfft,fs,b=25):
    # number of frequency bins
    nbins = nfft // 2 + 1
    # mel scale filter bank
    fbank = np.empty((b,nbins),dtype=float)
    # 50% overlap filter bank critical points (Hz)
    cp_f = _mel2hz(np.linspace(_hz2mel(f_min),_hz2mel(f_max),b+2))
    # get bin numbers for critical points (rounding down)
    cp_bin = (nfft * cp_f // fs).astype(int)
        
    # create mel-scale triangle filterbank
    for i,(fl,fc,fu) in enumerate(cp_bin[i:i+3] for i in range(len(cp_bin)-2)):
        # generator for filter gains
        gen = (float(k - fl) / (fc - fl) if k >= fl and k <= fc else 
               float(fu - k) / (fu - fc) if k > fc and k <= fu else 
               0 for k in range(nbins))
               
        fbank[i,:] = np.fromiter(gen,dtype=float)
    
    return fbank

# convert Hz to mel scale
def _hz2mel(freq):
    """Converts Hz to mel scale.
    
    Args:
        freq: Frequency in Hz
    """
    return 1127*np.log(1 + freq / 700.0)

# convert mel to Hz scale
def _mel2hz(freq):
    """Converts mel to Hz scale.
    
    Args:
        freq: Frequency in mels
    """
    return 700*(np.exp(freq / 1127.0) - 1)