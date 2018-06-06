from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import stat

from cv2 import imwrite, compareHist, HISTCMP_CHISQR_ALT
from math import log,sqrt
from videofeats import VideoFeats
from videostream import VideoStream
from google.cloud import storage
from google.cloud import videointelligence
from shutil import rmtree
from sklearn.metrics.pairwise import cosine_similarity
from subprocess import check_call
from timeit import default_timer
      
# function for generating keyframes from a skim of the video    
def kf_skim(vs,period=1,offset=0):
    """Generates a numpy array of keyframes using the skim of the video. 
    
    Args:
        vs:     VideoStream object
        period: Sampling period (in seconds) for grabbing frames. Default is 1
        offset: Starting point (in seconds) for grabbing frames. Default is 0
    """
    # video properties
    n = vs.frame_count
    fps = vs.fps
    # sampling period (in frame indices)
    ft = int(period * fps)
    
    # frame indices
    kf_ind = range(0+int(offset * fps),n,ft)
    
    return kf_ind
    
# distance-based method for selecting keyframes
def kf_dist(vs,return_kf=True):
    """ Generates a numpy array of keyframes using a distance based method.
    
    Args:
        vs: VideoStream object
    """
    # video properties
    n = vs.frame_count
    dim = (320,240) # processing dimensions (W x H)
    n_segs = (n-1 - 1) // 20 + 1
    
    # number of values to append
    if 0 < (n-1) % 20 < 10:
        n_app = 2
    elif (n-1) % 20 >= 10:
        n_app = 1
    else:
        n_app = 0
        
    #############################    
    #  First-round Differences  #
    #############################
        
    # frame indices
    frame_ind = np.pad(range(0,n,10),(0,n_app),'constant',constant_values=n-1) 
    
    # frame intensities
    Y = vs.frames(frame_ind,size=dim,colorspace='gray').astype(int)
    Y_diff = np.abs(np.diff(Y,axis=0))
    Y_diff2 = np.abs(np.diff(Y[::2,...],axis=0))
    
    # segment distances
    d  = np.fromiter((diff.sum() for diff in Y_diff2), dtype=int)
    df = np.fromiter((diff.sum() for diff in Y_diff[::2,...]), dtype=int)
    db = np.fromiter((diff.sum() for diff in Y_diff[1::2,...]), dtype=int)
    
    ###########################
    #  Threshold Computation  #
    ###########################
    
    # parameters
    p = 0.45
    m_g = d.mean()
    
    # local mean and standard deviation
    d_pad = np.pad(d,2,'constant')
    m_l = np.convolve(d_pad,0.2*np.ones(5),'valid')
    d_cent2 = np.pad((d-m_l)**2,2,'constant')
    var_l = np.convolve(d_cent2,0.25*np.ones(5),'valid')
    
    # local threshold
    t_l = np.fromiter((m_l[i] + p*sqrt(var_l[i])*(1 + log(m_g/m_l[i]))
                       if m_l[i] > 0 else 0 
                       for i in range(n_segs)),dtype=float,count=n_segs)
                                   
    #################################
    #  Candidate Segment Selection  #
    #################################

    # check which segments are above threshold
    test1 = d > t_l
    # check alternative test
    dm1 = np.append(0,d[:-1]) # d(n-1)
    dp1 = np.append(d[1:],0) # d(n+1)
    test2 = np.logical_and(np.logical_or(d > 3*dm1,d > 3*dp1), d > 0.8*m_g)
                                                                    
    # take union of test results and return segment indices
    psegs = np.where(np.logical_or(test1,test2))[0]
    
    ############################
    #  First-round Bisections  #
    ############################
    
    # first round segment distances
    d_f = d[psegs]
    df_f = df[psegs]
    db_f = db[psegs]
    
    # candidate CT in first 11 frames of segment (segment number specified)
    t1_f = psegs[np.logical_and(df_f > 1.5*db_f, df_f > 0.7*d_f)]
    # candidate CT in second 11 frames of segment
    t2_f = psegs[np.logical_and(db_f > 1.5*df_f, db_f > 0.7*d_f)]
    # false positive
    t3_f = psegs[np.logical_and(df_f < 0.3*d_f, db_f < 0.3*d_f)]
    # candidate GT in segment
    t4_f_ind = np.setdiff1d(psegs,np.hstack([t1_f,t2_f,t3_f]))*20
    
    ##############################
    #  Second-round Differences  #
    ############################## 
    
    # frame indices
    t1_f_ind = t1_f*20
    t2_f_ind = t2_f*20 + 10
    
    frame_ind = np.sort(np.hstack((t1_f_ind,t1_f_ind+5,t1_f_ind+10,
                                   t2_f_ind,t2_f_ind+5,t2_f_ind+10)))
    frame_ind[frame_ind > n-1] = n-1
    
    # checks if any potential CTs found. If none, then return no keyframes. This
    # should be updated at some point, this is usually the result of a video with
    # no transitions in it
    if frame_ind.size == 0:
        return np.array([])  
    
    # frame intensities
    Y = vs.frames(frame_ind,size=dim,colorspace='gray').astype(int)
    Y_diff = np.abs(np.diff(Y,axis=0))
    
    # segment distances
    d_s  = np.fromiter((np.abs(Y[i+1,...]-Y[i-1,...]).sum() 
                        for i in range(1,Y.shape[0]-1,3)), dtype=int)
    df_s = np.fromiter((diff.sum() for diff in Y_diff[::3,...]), dtype=int)
    
    db_s = np.fromiter((diff.sum() for diff in Y_diff[1::3,...]), dtype=int)
    
    #############################
    #  Second-round Bisections  #
    #############################
    
    # starting indices for candidate segments
    cs_ind = np.sort(np.append(t1_f_ind,t2_f_ind))
    
    # candidate CT in first 6 frames (frame index specified)
    t1_s_ind = cs_ind[np.logical_and(df_s > 1.5*db_s, df_s > 0.7*d_s)]
    # candidate CT in second 6 frames
    t2_s_ind = cs_ind[np.logical_and(db_s > 1.5*df_s, db_s > 0.7*d_s)]
    # false positive
    t3_s_ind = cs_ind[np.logical_and(df_s < 0.3*d_s, db_s < 0.3*d_s)]
    # candidate GT in segment
    t4_s_ind = np.setdiff1d(cs_ind,np.hstack([t1_s_ind,t2_s_ind,t3_s_ind]))
    
    ########################
    #  Candidate Segments  #
    ########################
    
    # Hard transitions (cuts)
    CT = np.sort(np.hstack([t1_s_ind,t2_s_ind+5]))
    # Gradual transitions (fades/dissolves)
    GT_long = t4_f_ind
    GT_short = np.sort(t4_s_ind)
    
    ########################
    #  Keyframe Selection  #
    ########################
    
    # candidate segments
    CT_segs = np.column_stack((CT,CT+5))
    GT_segs = np.row_stack((np.column_stack((GT_short,np.minimum(GT_short+10,n-1))),
                            np.column_stack((GT_long,np.minimum(GT_long+20,n-1)))))
    GT_segs = GT_segs[np.argsort(GT_segs[:,0]),:]
                            
    # merge overlapping GT segments
    merge_ind = np.where(np.diff(GT_segs.flatten()) <= 0)[0]
    merge_ind = np.hstack([merge_ind,merge_ind+1])
    GT_segs = np.delete(GT_segs.flatten(),merge_ind)
    GT_segs = GT_segs.reshape((len(GT_segs) // 2, 2))
    
    # form shot segments
    segs = np.row_stack((CT_segs,GT_segs))
    segs = segs[np.argsort(segs[:,0]),:]
    segs_d = [(seg[1] - seg[0]) // 5 for seg in segs]
    # flatten
    segs_fl = segs.flatten()
    # window around shot boundary
    segs_fl[::2] += segs_d
    segs_fl[1::2] -= segs_d
    # prepend 0 and append n-1
    segs = np.hstack((0,segs_fl,vs.frame_count-1))
    segs = segs.reshape(len(segs) // 2, 2)
    # find segments less than T frames and merge with closest segment
    T = int(0.6*vs.fps) # 0.6 second segment
    short = [i for i,seg in enumerate(segs) if seg[1]-seg[0]+1 < T]
    bdiffs = np.concatenate(([np.inf],np.diff(segs.flatten())[1::2],[np.inf]))
    # merge
    m_segs = segs.copy()
    for i in short:
        if bdiffs[i] < bdiffs[i+1]: # merge with left segment
            m_segs[i-1,] = [segs[i-1,0],segs[i,1]]
        else: # merge with right segment
            m_segs[i+1,] = [segs[i,0],segs[i+1,1]]
            
    m_segs = np.delete(m_segs,short,axis=0)
            
    
    # get keyframes (middle frame of each shot from unmerged segments)
    kf_ind = segs.mean(1).astype(int)
    
    # TODO: Make separate function for computing segments, don't leave as part 
    # this keyframe algorithm
    if return_kf:
        return list(kf_ind)
    else:
        return m_segs
    
# function for generating keyframes using Google Cloud Video Intelligence API
def kf_google(vs):
    # instantiate storage client and video client
    storage_client = storage.Client('adclass-1286')
    video_client = videointelligence.VideoIntelligenceServiceClient()
    
    # get video files bucket
    bucket = storage_client.get_bucket('video_files')
    
    # upload file to bucket
    blob = bucket.blob(vs.title)
    blob.upload_from_filename(vs.file)
    
    # construct video intelligence client request
    video_gcs = 'gs://video_files/' + blob.name
    features = [videointelligence.enums.Feature.SHOT_CHANGE_DETECTION]
    operation = video_client.annotate_video(video_gcs, features=features)
        
    # get results and delete object from bucket
    results = operation.result(timeout=90)
    blob.delete()
    
    # compute keyframe indices
    kf_ind = np.empty(0,dtype=int)
    
    for shot in results.annotation_results[0].shot_annotations:
        # start and end time of shot
        start_time = (shot.start_time_offset.seconds +
                      shot.start_time_offset.nanos / 1e9)
        end_time = (shot.end_time_offset.seconds +
                    shot.end_time_offset.nanos / 1e9)
        cur_ind = vs.fps * (start_time + end_time) / 2
        kf_ind = np.append(kf_ind,int(cur_ind))
    
    return list(kf_ind)

# pairwise comparison of histograms
def chi2_hist(X,Y=None):
    if Y is None:
        Y = X
      
    d = [[0.25*compareHist(h1,h2,HISTCMP_CHISQR_ALT) if i < j else 0
          for j,h1 in enumerate(Y)]
              for i,h2 in enumerate(X)]
    
    return np.array(d)

# function for adaptively selecting keyframes via submodular optimization
def kf_adaptive(vs,l1=1,l2=5,niter=500):
    # video properties
    n = vs.frame_count
    # subsample frames for first round dimensionality reduction using kf_dist
    # if dim_red:
    #     segs = kf_dist(vs,return_kf=False)
    #     if segs.size == 0:
    #         segs = [(0,n)] # fixes issues with videos containing no transitions
    # else:
    #     segs = [(0,n)]
      
    #################################
    #  adaptive keyframe selection  #
    #################################
    
    # get all frame indices
    # frame_inds = [range(*tuple(seg)) for seg in segs]
    
    # get histogram and HOG features
    vf = VideoFeats(vs.file,1024)
    feats_h = vf.hog
    feats_l = vf.labhist
    
    # # reference to current position in array
    # cur_pos = 0
        
    # pairwise comparisons of features
    w = cosine_similarity(feats_h)
    # is there a faster way of creating this matrix?
    #d = -0.5*additive_chi2_kernel(feats_l[cur_pos:cur_pos+n_seg,...])
    d = chi2_hist(feats_l)
    
    # computes objective function value for a given index set
    def objective(kf_ind):
        n = len(w)
        n_summ = len(kf_ind)
    
        # representativeness
        r = w[:,kf_ind].max(axis=0).sum()
        # uniqueness
        d_sub = d[kf_ind,kf_ind]
        u = 0
        for j in range(1,len(d_sub)):
            u += d[j,:j].min()
            
        return r + l1*u + l2*(n-n_summ)
        
    # keyframe index set
    best_kf_ind = []
    best_obj = 0
    objs = []
    kf_inds = []
    
    # submodular optimization
    for _ in range(niter):       
        # random permutation of frame indices
        u = np.random.permutation(n)
    
        # initial keyframe (relative) index sets
        # probably much faster with masked arrays
        X = []
        Y = list(u)
    
        for i,uk in enumerate(u):
            # remove uk from Y
            Y.remove(uk)
            
            # maximum similarity between each i and Y \ uk
            # minimum distance from uk to Y \ uk
            try:
                wY_max = np.amax(w[Y,],axis=0)
                dY_min = d[Y,uk].min()
            except ValueError:
                wY_max = np.zeros(n)
                dY_min = 0.0 # not sure how to define for empty set
                
            # maximum similarity between each i and X and 
            # minimum distance from uk to X
            try:
                wX_max = np.amax(w[X,],axis=0)
                dX_min = d[X,uk].min()
            except ValueError:
                wX_max = np.zeros(n)
                dX_min = 0.0 # not sure how to define for empty set
        
            # compute change in objective function
            df_X =   np.maximum(0,w[uk,]-wX_max).sum() + l1*dX_min - l2
            df_Y = -(np.maximum(0,w[uk,]-wY_max).sum() + l1*dY_min - l2)
        
            # probabilistically add/remove elements to/from X/Y
            a = max(df_X,0.0)
            b = max(df_Y,0.0)      
            if a + b == 0.0:
                prob_X = 0.5
            else:
                prob_X = a / (a + b)
        
            if np.random.uniform() < prob_X:
                X.append(uk)
                # put uk back in Y
                Y.append(uk)
                
        # update keyframe set
        kf_ind = sorted(X)
        obj = objective(kf_ind)
        objs.append(obj)
        kf_inds.append(kf_ind)
        if obj > best_obj:
            best_kf_ind = kf_ind
            best_obj = obj
    
    return best_kf_ind
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file', 
                        help='Path to video file for keyframe extraction')
    args = parser.parse_args()
    # get arguments
    video_file = args.video_file
    # resize video to 320 x 240
    fn,ext = os.path.splitext(video_file)
    resized_file = fn + '_res' + ext
    
    cmd = ['ffmpeg','-y','-i',video_file,'-vf','scale=320:240',resized_file]
    check_call(cmd)
    
    # directory name of file
    dir_name = os.path.dirname(video_file)
    # output directory name
    out_dir = os.path.join(dir_name,'summary')        
    # remove output directory if already in existence
    if os.path.exists(out_dir):
        os.chmod(out_dir, stat.S_IWUSR) # grant all privileges
        rmtree(out_dir,ignore_errors=True)
        
    # creat output directory
    try:
        os.mkdir(out_dir)
    except WindowsError: # occurs when folder is open in explorer when running
        os.mkdir(out_dir)
        
    
    # instantiate VideoStream object
    vs_res = VideoStream(resized_file)
    vs_orig = VideoStream(video_file)
    
    # skim keyframes
    t1 = default_timer()
    ind_s = kf_skim(vs_res)
    kf_s = vs_orig.frames(ind_s)
    t_s = default_timer() - t1
    # distance keyframes
    t1 = default_timer()
    ind_d = kf_dist(vs_res)
    kf_d = vs_orig.frames(ind_d)
    t_d = default_timer() - t1
    # Google keyframes
    t1 = default_timer()
    ind_g = kf_google(vs_res)
    kf_g = vs_orig.frames(ind_g)
    t_g = default_timer() - t1
    # adaptive keyframes
    t1 = default_timer()
    ind_a = kf_adaptive(vs_res,l2=6,dim_red=True)
    kf_a = vs_orig.frames(ind_a)
    t_a = default_timer() - t1
    
    # write keyframes to output directory
    skim_dir = os.path.join(out_dir,'skim')
    os.mkdir(skim_dir)  
    for i,kf in enumerate(kf_s):
        # reject monochramatic frames (determined by intensity variance)
        if np.average(kf,axis=2,weights=[0.114,0.587,0.299]).std() >= 10:
            imwrite(os.path.join(skim_dir,'frame_{0:04d}.png'.format(ind_s[i])),kf)
            
    dist_dir = os.path.join(out_dir,'dist')
    os.mkdir(dist_dir)
    for i,kf in enumerate(kf_d):
        # reject monochramatic frames (determined by intensity variance)
        if np.average(kf,axis=2,weights=[0.114,0.587,0.299]).std() >= 10:
            imwrite(os.path.join(dist_dir,'frame_{0:04d}.png'.format(ind_d[i])),kf)
            
    goog_dir = os.path.join(out_dir,'google')        
    os.mkdir(goog_dir)
    for i,kf in enumerate(kf_g):
        # reject monochramatic frames (determined by intensity variance)
        if np.average(kf,axis=2,weights=[0.114,0.587,0.299]).std() >= 10:
            imwrite(os.path.join(goog_dir,'frame_{0:04d}.png'.format(ind_g[i])),kf)
            
    adap_dir = os.path.join(out_dir,'adaptive')        
    os.mkdir(adap_dir)
    for i,kf in enumerate(kf_a):
        # reject monochramatic frames (determined by intensity variance)
        if np.average(kf,axis=2,weights=[0.114,0.587,0.299]).std() >= 10:
            imwrite(os.path.join(adap_dir,'frame_{0:04d}.png'.format(ind_a[i])),kf)
    
    # delete resized video file
    os.remove(resized_file)
            
    # print out runtimes
    print('\n')
    print('kf_skim() runtime: {0:04f}\n'.format(t_s))
    print('kf_dist() runtime: {0:04f}\n'.format(t_d))
    print('kf_google() runtime: {0:04f}\n'.format(t_g))
    print('kf_adaptive() runtime: {0:04f}\n'.format(t_a))