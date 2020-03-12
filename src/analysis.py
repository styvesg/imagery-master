import sys
import os
import struct
import time
import numpy as np
import h5py
from glob import glob
import nibabel as nib
import scipy.io as sio
from scipy import ndimage as nd
from scipy.io import loadmat
from scipy import misc
from scipy.stats import pearsonr
from tqdm import tqdm
import pickle
import math

def _or(a,b):
    return np.logical_or(a,b)
def _and(a,b):
    return np.logical_and(a,b)

def log_scale_intervals(x):
    a = (np.log(x)[-1]-np.log(x)[0])/len(x)
    b = np.log(x)[0]
    dx = np.array([(np.exp(b+(i+0.5)*a) - np.exp(b+(i-0.5)*a)) for i in range(len(x))])
    return dx

def dist_to_parity(x, y): 
    '''There would be an additional factor of 1/sqrt(2) if we want to measure the shortest distance to from a point (x,y) to the parity line y=x.'''
    return y-x  

def pretty_accuracy_fit(X, Y): 
    '''minimize the square of the shortest distance from a point to a line -- this is different than the normal least square fit, which minimizes the residual.
    The problem with the latter is that it doesn't handle the constraint b=0 very well and heavy discount points above the origin such that, even if one had a perfectly
    symmetric distribution of point, one would not have a fit with a=1. Minimizing the shortest distance from points to line solves this issue, and we recover a=1 for
    symmetric distributions. The following is based on explicitly solving the ensuing variational problem.'''
    c = np.sum(X*Y)
    b = np.sum(Y*Y)-np.sum(X*X)
    return (b+np.sqrt(b**2+4*c**2))/(2*c)

def pretty_accuracy_fit_error(X, Y):
    e = np.std(Y-X)
    c = np.sum(X*Y)
    b = np.sum(Y*Y)-np.sum(X*X)
    return (b+np.sqrt(b**2+4*c**2))/(2*c), e

def iterate_roi(group, voxelroi, roimap, group_name=None):
    for k,g in enumerate(group):
        g_name = ('' if group_name is None else group_name[k])
        mask = np.zeros(shape=voxelroi.shape, dtype=bool)
        for i,roi in enumerate(g):
            if group_name is None:
                g_name += roimap[roi] + ('-' if i+1<len(g) else '')
            mask = np.logical_or(mask, voxelroi==roi)
        yield mask, g_name
        

import nibabel as nib
def load_mask_from_nii(mask_nii_file):
    return nib.load(mask_nii_file).get_data()
    
def view_data(vol_shape, idx_mask, data_vol, save_to=None):
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    view_vol[idx_mask.astype('int').flatten()] = data_vol
    view_vol = view_vol.reshape(vol_shape, order='F')
    if save_to:
        nib.save(nib.Nifti1Image(view_vol, affine=np.eye(4)), save_to)
    return view_vol

def show_valid_only(x, mask):
    y = np.copy(x)
    y[mask] = np.nan
    return y


def sample_tuning(ri, rc, cc): # returns sample_fi[sample, freq, voxel]
    return np.square(rc) / np.sum(np.square(rc), axis=1, keepdims=True) #  / np.square(cc)[:,np.newaxis,:]
    #return ri / cc[:,np.newaxis,:]


def sample_tuning_avg(freq, sample_fi): # returns sample_fi_avg[sample, voxel]
    return np.sum(sample_fi*freq[np.newaxis,:,np.newaxis], axis=1)
def sample_roi_tuning(sample_fi): # returns sample_Fi[sample, freq]
    return np.mean(sample_fi, axis=2)
def sample_roi_tuning_avg(sample_fi_avg): # returns sample_Fi_avg[sample]    
    return np.mean(sample_fi_avg, axis=1)    
def tuning_from_sample(sample_fi):
    return np.mean(sample_fi, axis=0), np.std(sample_fi, axis=0)
def tuning_avg_from_sample(sample_fi_avg):
    return np.mean(sample_fi_avg, axis=0), np.std(sample_fi_avg, axis=0)

    
def tuning(ri, dri, rc, drc, cc, dcc):
    rc2  = np.square(rc)
    src2 = np.sum(rc2, axis=0, keepdims=True) 
    # fi = ri / cc[np.newaxis,:]
    fi = rc2 / src2 
    dfi = fi* (2*drc / rc + 4*np.sum(np.square(rc * drc), axis=0, keepdims=True) / src2)    
    return fi, dfi
def tuning_avg(freq, fi, dfi):
    return np.sum(fi*freq[:,np.newaxis], axis=0), np.sqrt(np.sum(np.square(freq[:,np.newaxis]*dfi), axis=0))


def roi_tuning(fi, dfi, weights=None):
    if weights is None:
        return np.mean(fi, axis=1), np.sqrt(np.sum(np.square(dfi), axis=1)) / fi.shape[1]
    else:
        return np.average(fi, weights=weights, axis=1), np.sqrt(np.sum(np.square(dfi) * weights, axis=1)) / (fi.shape[1] * np.sum(weights))


def roi_tuning_avg(freq, Fi, dFi):
    return np.sum(Fi*freq[:,np.newaxis], axis=0), np.sqrt(np.sum(np.square(freq[:,np.newaxis]*dFi), axis=0))


def weighted_average(weight, value):
    sw = np.sum(weight)
    w = weight / sw
    return np.sum(w*value)


def evaluate_tuning(models, parts, conditions):
    tuning_dicts = {}
    for m in conditions:
        tuning_dicts[m] = {}
        model = models[m]
        part  = parts[m]
        ## voxel-wise
        val_cc = np.mean(np.nan_to_num(part['val_cc']), axis=0)
        cc_std = np.std(np.nan_to_num(part['val_cc']), axis=0)
        val_ri = np.mean(np.nan_to_num(part['val_ri']), axis=0)
        ri_std = np.std(np.nan_to_num(part['val_ri']), axis=0)
        val_rc = np.mean(np.nan_to_num(part['val_rc']), axis=0)
        rc_std = np.std(np.nan_to_num(part['val_rc']), axis=0)       
        idx_sort = np.argsort(val_cc)[::-1] ## best to worse

        tuning_dicts[m]['idx_sorted'] = idx_sort
        tuning_dicts[m]['val_cc_sorted'] = val_cc[idx_sort]
        tuning_dicts[m]['cc_std_sorted'] = cc_std[idx_sort] 
        tuning_dicts[m]['val_cc'] = val_cc
        tuning_dicts[m]['cc_std'] = cc_std    

        tuning_dicts[m]['sample_fi'] = sample_fi = sample_tuning(\
            np.nan_to_num(part['val_ri']), np.nan_to_num(part['val_rc']), np.nan_to_num(part['val_cc']))
        tuning_dicts[m]['fi'], tuning_dicts[m]['dfi'] = tuning_from_sample(sample_fi)
    return tuning_dicts


def evaluate_roi_tuning(tuning_dicts, masks, conditions, roi_group, voxel_roi, roi_map):
    for m in conditions:
        ## ROI all
        cc_mask = masks[m]   
        partition_R_avg = np.ndarray(shape=(tuning_dicts[m]['sample_fi'].shape[1], len(roi_group)), dtype=tuning_dicts[m]['sample_fi'].dtype)
        partition_R_std = np.ndarray(shape=(tuning_dicts[m]['sample_fi'].shape[1], len(roi_group)), dtype=tuning_dicts[m]['sample_fi'].dtype)
        for roi, value in enumerate(iterate_roi(roi_group, voxel_roi.flatten(), roi_map)):
            roi_mask, name = value
            mask = np.logical_and(roi_mask, cc_mask)
            partition_R_avg[:,roi], partition_R_std[:,roi] = roi_tuning(*tuning_from_sample(tuning_dicts[m]['sample_fi'][:,:,mask]))        
        tuning_dicts[m]['Fi_all'] = partition_R_avg
        tuning_dicts[m]['dFi_all'] = partition_R_std 
        ## ROI pic
        cc_mask = masks['pic'] 
        partition_R_avg = np.ndarray(shape=(tuning_dicts[m]['sample_fi'].shape[1], len(roi_group)), dtype=tuning_dicts[m]['sample_fi'].dtype)
        partition_R_std = np.ndarray(shape=(tuning_dicts[m]['sample_fi'].shape[1], len(roi_group)), dtype=tuning_dicts[m]['sample_fi'].dtype)
        for roi, value in enumerate(iterate_roi(roi_group, voxel_roi.flatten(), roi_map)):
            roi_mask, name = value
            mask = np.logical_and(roi_mask, cc_mask)
            partition_R_avg[:,roi], partition_R_std[:,roi] = roi_tuning(*tuning_from_sample(tuning_dicts[m]['sample_fi'][:,:,mask]))        
        tuning_dicts[m]['Fi_pic'] = partition_R_avg
        tuning_dicts[m]['dFi_pic'] = partition_R_std
    return tuning_dicts

################# PREDICTION ACCURACY ####################
def resample_cc(X):
    cc = np.ndarray(shape=X.shape[1:])
    choice = np.random.randint(len(X), size=(X.shape[1]))
    for v in range(X.shape[1]):
        cc[v] = X[choice[v],v]
    return cc

def get_cc_sample(X, Y):  
    return resample_cc(X), resample_cc(Y)

def get_cc_resampling(cX, cY, n_samples=1000, randomize=False):
    parity = []
    if randomize:
        for s in range(n_samples):
            swap = np.random.randint(0,2, size=cX.shape[1]).astype(np.bool)
            X, Y  = np.copy(cX), np.copy(cY)
            X[:,swap], Y[:,swap] = cY[:,swap], cX[:,swap]
            x, y = get_cc_sample(X, Y)
            parity += [pretty_accuracy_fit(x, y)-1.0,] # [dist_to_parity(np.mean(x), np.mean(y)),] 
    else:
        for s in range(n_samples):
            x, y = get_cc_sample(cX, cY)
            parity += [pretty_accuracy_fit(x, y)-1.0,] # [dist_to_parity(np.mean(x), np.mean(y)),] 
    return parity

################# RF ECCENTRICITY ####################
def resample_rf(X):
    rf = np.ndarray(shape=X.shape[1:])
    choice = np.random.randint(len(X), size=(X.shape[1]))
    for v in range(X.shape[1]):
        rf[v] = X[choice[v],v]
    return rf

def get_ecc_sample(X, Y):  
    x, y = resample_rf(X), resample_rf(Y)
    x_ecc = np.sqrt(np.square(x[:,0]) + np.square(x[:,1]))
    y_ecc = np.sqrt(np.square(y[:,0]) + np.square(y[:,1]))
    return x_ecc, y_ecc

def get_ecc_resampling(X, Y, n_samples=1000, randomize=False):
    parity = []
    if randomize:
        for s in range(n_samples):
            swap = np.random.randint(0,2, size=X.shape[1]).astype(np.bool)
            x, y  = np.copy(X), np.copy(Y)
            x[:,swap], y[:,swap] = Y[:,swap], X[:,swap]
            x_ecc, y_ecc = get_ecc_sample(x, y)
            parity += [dist_to_parity(np.mean(x_ecc), np.mean(y_ecc)),]
    else:
        for s in range(n_samples):
            x_ecc, y_ecc = get_ecc_sample(X, Y)
            parity += [dist_to_parity(np.mean(x_ecc), np.mean(y_ecc)),]
    return parity

#################### RF SIZE ######################
def get_size_sample(X, Y):
    return resample_rf(X), resample_rf(Y)

def get_size_resampling(X, Y, n_samples=1000, randomize=False):
    avg_size = []
    if randomize:
        for s in range(n_samples):
            swap = np.random.randint(0,2, size=X.shape[1]).astype(np.bool)
            x, y  = np.copy(X), np.copy(Y)
            x[:,swap], y[:,swap] = Y[:,swap], X[:,swap]
            x_size, y_size = get_size_sample(x, y)
            avg_size += [dist_to_parity(np.mean(x_size), np.mean(y_size)),]
    else:
        for s in range(n_samples):
            x_size, y_size = get_size_sample(X, Y)
            avg_size += [dist_to_parity(np.mean(x_size), np.mean(y_size)),]
    return avg_size

################# FREQUENCY TUNING ####################
def resample_tuning(X):
    fi_avg = np.ndarray(shape=X.shape[1:])
    choice = np.random.randint(len(X), size=(X.shape[1]))
    for v in range(X.shape[1]):
        fi_avg[v] = X[choice[v],v]
    return fi_avg

def get_tuning_sample(X, Y):
    return resample_tuning(X), resample_tuning(Y)

def get_tuning_resampling(X, Y, n_samples=1000, randomize=False):
    avg_f = []
    if randomize:
        for s in range(n_samples):
            swap = np.random.randint(0,2, size=X.shape[1]).astype(np.bool)
            x, y  = np.copy(X), np.copy(Y)
            x[:,swap], y[:,swap] = Y[:,swap], X[:,swap]
            x_f, y_f = get_tuning_sample(x, y)
            avg_f += [dist_to_parity(np.mean(x_f), np.mean(y_f)),]
    else:
        for s in range(n_samples):
            x_f, y_f = get_tuning_sample(X, Y)
            avg_f += [dist_to_parity(np.mean(x_f), np.mean(y_f)),]
    return avg_f
