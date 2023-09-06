import numpy as np
import h5py
from glob import glob
import PIL.Image as pim
import pickle


def mask_grayscale_cue(stim):
    npx = stim.shape[2]
    xstart,xstop = int(np.floor(266.*npx/600)), int(np.ceil(334.*npx/600))
    ystart,ystop = int(np.floor(291.*npx/600)), int(np.ceil(310.*npx/600))
    msk_stim = np.copy(stim)
    msk_stim[:,:,ystart:ystop, xstart:xstop] = 0.58039218 # hardcoded gray value
    return msk_stim


def load_stimuli(path, load_data=True):
    data = None
    if load_data:
        image_h5 = h5py.File(path + "Stimuli.h5py",'r')
        data = np.array(image_h5['stim'])
        image_h5.close()
    meta_file = open(path + "Stimuli_metadata.pkl", 'rb')
    meta = pickle.load(meta_file)
    meta_file.close()
    return data, meta

def load_grayscale_stimuli(path, npx):
    data, meta = load_stimuli(path)
    return grayscale_stimuli(data, npx), meta



def load_voxels(path, load_data=True):
    data = None
    if load_data:
        voxel_h5 = h5py.File(path + "Voxels.h5py",'r')
        data = np.array(voxel_h5['betas'])
        voxel_h5.close()
    meta_file = open(path + "Voxels_metadata.pkl", 'rb')
    meta = pickle.load(meta_file)
    meta_file.close()
    return data, meta

def load_zscored_voxels(path):
    data, meta = load_voxels(path)
    return zscore_voxels(data), meta



def split_mask(stimuli_meta):
    imask = np.arange(len(stimuli_meta['seq']), dtype=int)[np.array(stimuli_meta['seq'])=='img']
    pmask = np.arange(len(stimuli_meta['seq']), dtype=int)[np.array(stimuli_meta['seq'])=='pcp']
    # sanity check
    assert np.prod(np.array(stimuli_meta['clo'])[imask]==np.array(stimuli_meta['clo'])[pmask]).astype(bool),\
        "The ordering of the objects differs between the imagery and perception sets"
    assert np.prod(np.array(stimuli_meta['ori'])[imask]==np.array(stimuli_meta['ori'])[pmask]).astype(bool),\
        "The ordering of the orientations differs between the imagery and perception sets"
    return pmask, imask


def grayscale_stimuli(data, npx):
    gray_data = np.ndarray(shape=(len(data), npx, npx), dtype=np.float32)
    for i,rawim in enumerate(data):
        im = pim.fromarray(rawim, mode='RGB').resize((npx, npx), resample=pim.BILINEAR).convert('L')
        gray_data[i,...] = np.asarray(im).astype(np.float32)      
    gray_data /= 255
    return gray_data[:,np.newaxis]


def zscore_voxels(data):
    a = np.mean(data, axis=0, keepdims=True)
    s = np.std(data, axis=0, keepdims=True)
    z = (data - a) / s
    return np.nan_to_num(z)


def flatten_dict(base, append=''):
    flat = {}
    for k,v in base.items():
        if type(v)==dict:
            flat.update(flatten_dict(v, append+k+'.'))
        else:
            flat[append+k] = v
    return flat

def embbed_dict_element(d, k, v):
    ks = k.split('.')
    if len(ks)>1:
        if ks[0] not in d:
            d[ks[0]] = {}
        d[ks[0]] = embbed_dict_element(d[ks[0]], '.'.join(ks[1:]), v)
    else:
        d[k] = v            
    return d

def embbed_dict(fd):
    d = {}
    for k in fd.keys():
        d = embbed_dict_element(d, k, fd[k])
    return d
