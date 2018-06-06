import cv2
import numpy as np
import skimage.measure as skm

# text detection in an image based on MSER
def text_detect(im,delta=4):
    """ For a given image, computes the number of maximally stable extremal
    regions according to several regions properties (eccentricity, aspect ratio,
    solidity, extent, Euler number).
    
    Args:
        im:    Numpy array representing the image (RGB or grayscale)
        delta: Step size between intensity threshold levels, specified as a
               percent of the maximum intensity (typical values range from 0.8 -
               4)
    """
    # convert to grayscale if necessary
    if len(im.shape) != 2:
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    # image properties
    im_size = im.shape # H x W
        
    # MSER detector parameters
    delta = int(round(delta*255/100.0))
    # region sizes
    min_area = 200
    max_area = 8000
    # area variation
    area_var = 0.25
    
    # construct MSER object
    mser = cv2.MSER_create(_delta=delta,_min_area=min_area,_max_area=max_area,
                           _max_variation=area_var)
                           
    # detect regions
    regions, _ = mser.detectRegions(im)
    
    # plot regions before filtering
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(im, hulls, 1, (0, 255, 0))
    cv2.imshow('Pre-filtered Regions', im)
    
    # generate labeled image
    lab_im = np.zeros(im_size,dtype=int) # H x W
    for cc_num, region in enumerate(regions):
        # better way to do this without loop?
        for px, py in region:
            lab_im[py,px] = cc_num + 1 # 1-based indexing for CC
            
    # conncected component properties
    mser_stats = skm.regionprops(lab_im)
    
    # threshold-based region filtering
    regions_f = []
    num_regions = 0
    for i,prop in enumerate(mser_stats):
        # compute aspect ratio
        aspect = float(prop.bbox[3] - prop.bbox[1]) / (prop.bbox[2] - prop.bbox[0])
        # filter step
        if (aspect <= 3 and prop.eccentricity <= 0.995 and prop.solidity >= 0.3
            and 0.2 <= prop.extent <= 0.9 and prop.euler_number >= -4):
            regions_f.append(regions[i])
            num_regions = num_regions + 1
            
    # plot regions after filtering
    hulls_f = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions_f]
    cv2.polylines(im, hulls_f, 1, (0, 255, 0))
    cv2.imshow('Post-filtered Regions', im)
            
    return num_regions