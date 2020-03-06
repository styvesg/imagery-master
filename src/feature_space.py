import importlib

import numpy as np
import numpy_utility as pnu
from fwrf import fpX
gaborizer = importlib.import_module("gaborizer.src.gabor_feature_dictionaries")


def preprocess_gabor_feature_maps(feat_dict, act_func=None, dtype=np.float32):
    '''
    Apply optional nonlinearity to the feature maps itself and concatenate feature maps of the same dimensions.
    Returns the feature maps and a list of theano variables to represent them, and the shape of the fmaps.
    '''
    fmap_rez = []
    for k in feat_dict.keys():
        fmap_rez += [feat_dict[k].shape[2],]
    resolutions = np.unique(fmap_rez)
    # concatenate and sort as list
    fmaps_res_count = len(resolutions)
    fmaps_count = 0
    fmaps, _fmaps = [], []
    for r in range(fmaps_res_count):
        fmaps  += [[],]
    nonlinearity = act_func
    if nonlinearity is None:
        nonlinearity = lambda x: x
    for k in feat_dict.keys():
        # determine which resolution idx this map belongs to
        ridx = np.argmax(resolutions==feat_dict[k].shape[2])
        if len(fmaps[ridx])==0:
            fmaps[ridx] = nonlinearity(feat_dict[k].astype(dtype))
        else:
            fmaps[ridx] = np.concatenate((fmaps[ridx], nonlinearity(feat_dict[k].astype(dtype))), axis=1)       
        fmaps_count += 1
    fmaps_sizes = [] 
    for fmap in fmaps:
        fmaps_sizes += [fmap.shape]
    print fmaps_sizes
    print "total fmaps = %d" % fmaps_count 
    return fmaps, fmaps_sizes 



def create_gabor_feature_maps(stim_data, gabor_params, nonlinearity=lambda x: x):
    '''input should be a dictionary of control parameters
    output should be the model space tensors'''
    print gabor_params
    n_ori = gabor_params['n_orientations']
    gfm = gaborizer.gabor_feature_maps(n_ori,\
        gabor_params['deg_per_stimulus'], (gabor_params['lowest_sp_freq'], gabor_params['highest_sp_freq'], gabor_params['num_sp_freq']),\
        pix_per_cycle=gabor_params['pix_per_cycle'], complex_cell=gabor_params['complex_cell'],\
        diams_per_filter = gabor_params['diams_per_filter'],\
        cycles_per_radius = gabor_params['cycles_per_radius'])
    #
    fmaps, fmaps_sizes = preprocess_gabor_feature_maps(gfm.create_feature_maps(stim_data), act_func=nonlinearity, dtype=fpX)
    fmaps_res_count = len(fmaps_sizes)
    fmaps_count = sum([fm[1] for fm in fmaps_sizes])
    #
    ori  = np.array(gfm.gbr_table['orientation'])[0:fmaps_count:fmaps_res_count] # leapfrog
    freq = np.array(gfm.gbr_table['cycles per deg.'])[:fmaps_res_count]
    env  = np.array(gfm.gbr_table['radius of Gauss. envelope (deg)'])[:fmaps_res_count]
    # preprocess_gabor_feature_maps sorts the frequencies and angle such that all feature with the same freq are contiguous.
    partitions = [0,]
    for r in fmaps_sizes:
        partitions += [partitions[-1]+r[1],]
    freq_rlist  = [range(start,stop) for start,stop in zip(partitions[:-1], partitions[1:])] # the frequency ranges list
    ori_rlist = [range(0+i,partitions[-1],n_ori) for i in range(0,n_ori)] # the angle ranges list
    return fmaps, freq, env, ori, freq_rlist, ori_rlist, fmaps_sizes, fmaps_count


