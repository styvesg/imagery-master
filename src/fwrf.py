########################################################################
### PACKAGE VERSIONS:												 ###
### theano: 	0.9  												 ###
### lasagne: 	0.2dev1												 ###
### numpy:
###                               
########################################################################

import sys
import struct
import time
import numpy as np
from tqdm import tqdm
import pickle
import math

import theano
import theano.tensor as T
import lasagne

import numpy_utility as pnu
import lasagne_utility as plu


fpX = np.float32
print "theano floatX: %s" % theano.config.floatX
print "numpy floatX: %s" % fpX

########################################################################
###              SUPPORT FUNCTIONS                                   ###
########################################################################

def flat_init(shape):
    return np.full(shape=shape, fill_value=1./np.prod(shape[1:]), dtype=fpX)
def zeros_init(shape):
    return np.zeros(shape=shape, dtype=fpX)
def ones_init(shape):
    return np.ones(shape=shape, dtype=fpX)
class normal_init(object):
    def __init__(self, scale=1.):
        self.scale = scale   
    def __call__(self, shape):
        return np.random.normal(0, self.scale, size=shape).astype(fpX)

class subdivision_1d(object):
    def __init__(self, n_div=1, dtype=fpX):
        self.length = n_div
        self.dtype = dtype
        
    def __call__(self, center, width):
        '''	returns a list of point positions '''
        return [center] * self.length
    
class linspace(subdivision_1d):    
    def __init__(self, n_div, right_bound=False, dtype=fpX, **kwargs):
        super(linspace, self).__init__(n_div, dtype=fpX, **kwargs)
        self.__rb = right_bound
        
    def __call__(self, center, width):
        if self.length<=1:
            return [center]     
        if self.__rb:
            d = fpX(width)/(self.length-1)
            vmin, vmax = center, center+width  
        else:
            d = fpX(width)/self.length
            vmin, vmax = center+(d-width)/2, center+width/2 
        return np.arange(vmin, vmax+1e-12, d).astype(dtype=self.dtype)
    
class logspace(subdivision_1d):    
    def __init__(self, n_div, dtype=fpX, **kwargs):
        super(logspace, self).__init__(n_div, dtype=fpX, **kwargs)
               
    def __call__(self, start, stop):    
        if self.length <= 1:
            return [start]
        lstart = np.log(start+1e-12)
        lstop = np.log(stop+1e-12)
        dlog = (lstop-lstart)/(self.length-1)
        return np.exp(np.arange(lstart, lstop+1e-12, dlog)).astype(self.dtype)

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual

def iterate_bounds(start, length, batchsize):
    batch_count = int(length // batchsize)
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield [start+i*batchsize, start+(i+1)*batchsize], batchsize
    if(residual>0):
        yield [start+batch_count*batchsize, start+length], residual	

def iterate_slice(start, length, batchsize):
    batch_count = int(length // batchsize)
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield slice(start+i*batchsize, start+(i+1)*batchsize), batchsize
    if(residual>0):
        yield slice(start+batch_count*batchsize,start+length), residual
        
def slice_arraylist(inputs, excerpt):            
    return [i[excerpt] for i in inputs]  

def iterate_array(inputs, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    for start_idx in range(0, len(inputs), batchsize):
        excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt]

def iterate_minibatches(inputs, targets, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs), batchsize):
        excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]
        
def iterate_multiminibatches(inputs, targets, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    length = len(targets)
    batch_count = len(targets) // batchsize 
    residual = length % batchsize    
    for start_idx in range(0, length-residual, batchsize):
        excerpt = slice(start_idx, start_idx+batchsize)
        yield [i[excerpt] for i in inputs] + [targets[excerpt]]
    if(residual>0):
        excerpt = slice(length-residual, length)
        yield [i[excerpt] for i in inputs] + [targets[excerpt]]


def unique_rel_to_abs_models(rel_models, rx, ry, rs):
    '''converts a list of relative models to the absolute model specified by the range parameters rx, ry, rs
    returns a matrix of size (number of models, 3)
    '''
    nv = len(rel_models)
    nx, ny, ns = len(rx[1]), len(ry[1]), len(rs[1])
    assert nv==len(rx[0])
    ixs, iys, iss = np.unravel_index(rel_models, (nx, ny, ns))
    abs_models = np.ndarray(shape=(nv, 3), dtype=fpX)
    for v in range(nv):
        abs_models[v] = [rx[v,ixs[v]], ry[v,iys[v]], rs[v,iss[v]]]
    return abs_models

def rel_to_abs_shared_models(rel_models, rx, ry, rs):
    '''converts a list of relative models to the absolute model specified by the range parameters rx, ry, rs
    returns a matrix of size (number of models, 3)'''
    nv = len(rel_models)
    nx, ny, ns = len(rx), len(ry), len(rs)
    ixs, iys, iss = np.unravel_index(rel_models, (nx, ny, ns))
    return np.stack([rx[ixs[:]], ry[iys[:]], rs[iss[:]]], axis=1)


def pRF(fwrf_weights, fmap_rf, pool_rf):
    '''
    fwrf_weights is [nv, nf]
    fmap_rf is [nf] i.e. it specifies a gaussian sigma value for each feature map
    pool_rf is [nv, 3] i.e. it specifies a gaussian population pooling fct for each voxel 
    
    returns [nv,3], a rf x, y, and sigma for each voxel
    '''
    # we'd be better off performing the implicit convolution first.
    vsigma = np.zeros(shape=(nv), dtype=fpX)
    for v in pool_rf:
        vsigma[v] = np.average(np.sqrt(np.square(fmap_rf) + np.square(pool_rf[v,2,np.newaxis])), weights=fwrf_weights[v,:])
    return np.stack([pool_rf[:,0:1], pool_rf[:,1:2], vsigma[:,np.newaxis]], axis=1)

########################################################################
###                                                                  ###
########################################################################

def create_shared_batched_feature_maps_gaussian_weights(fmap_sizes, batch_v, batch_t, verbose=True):
    nf = 0
    _smsts = []
    mem_approx = 0
    rep_approx = 0
    for i,a in enumerate(fmap_sizes):
        nf += a[1]
        n_pix = a[2]
        assert n_pix==a[3], "Non square feature map not supported"
        _smsts += [theano.shared(np.zeros(shape=(batch_v, batch_t, n_pix, n_pix), dtype=fpX)),]
        mem_approx += 4*batch_v*batch_t*n_pix**2
        rep_approx += 4*a[1]*n_pix**2
        if verbose:
            print "> feature map candidates %d with shape %s" % (i, (batch_v, batch_t, n_pix, n_pix))
    if verbose:        
        print "  total number of feature maps = %d, in %d layers" % (nf, len(fmap_sizes))
        print "  feature map candidate using approx %.1f Mb of memory (VRAM and RAM)" % (fpX(mem_approx) /(1024*1024))
    return _smsts, nf

    
def set_shared_batched_feature_maps_gaussian_weights(_psmsts, xs, ys, ss, env=None, size=20.):
    '''
    The interpretation of receptive field weight factor is that they correspond, for each voxel, to the probability of this voxel of seeing 
    (through the weighted average) a given feature map pixel through its receptive field size and position in visual space. 
    Whether that feature map pixel is relevant to the representation of that particular voxel is left to the voxel encoding model to decide.
    '''
    nf = 0
    (nv, nt) = (len(xs), 1) if xs.ndim==1 else xs.shape[0:2]
    (sv, st) = _psmsts[0].get_value().shape[0:2]
    assert nv==sv and nt==st, "non conformal (%d,%d)!=(%d,%d)" % (nv, nt, sv, st)
    for i,a in enumerate(_psmsts):
        n_pix = a.get_value().shape[2]
        ##
        sigmas = ss.flatten()
        if env is not None:
            sigmas = np.sqrt(np.square(sigmas) - np.square(env[i]))
            sigmas[np.isnan(sigmas)] = 0
        ##
        _,_,mst = pnu.make_gaussian_mass_stack(xs.flatten(), ys.flatten(), sigmas, n_pix, size=size, dtype=fpX)
        a.set_value(mst.reshape((nv, nt, n_pix, n_pix)))
    return _psmsts



def set_shared_parameters(shared_vars, values=None):
    if values:
        for var, val in zip(shared_vars, values):
            var.set_value(val.astype(fpX).reshape(var.get_value(borrow=True, return_internal_type=True).shape))    
    else: #clear vram
        for var in shared_vars:
            var.set_value(np.asarray([0,], dtype=fpX).reshape((1,)*len(var.get_value(borrow=True, return_internal_type=True).shape)))



def make_batched_regression(_mst_data, nf, nv, nc, add_bias=True):
    _W = theano.shared(np.zeros(shape=(nv, nc, nf), dtype=fpX))
    ### place voxel-candidate as the first dimension to be batched over.
    _pred = T.batched_tensordot(_mst_data.flatten(ndim=3).dimshuffle((2,0,1)), \
        _W.reshape((nv*nc, nf)), axes=[[2],[1]]) \
        .dimshuffle((1,0)).reshape((_mst_data.shape[0],nv,nc))
    params = [_W,]
    if add_bias:
        _b = theano.shared(np.zeros(shape=(1, nv, nc), dtype=fpX))
        _pred = _pred + T.patternbroadcast(_b, (True, False, False))
        params += [_b,]
    return _pred, params

def make_mst_data(_fmaps, _smsts): 
    '''Apply a tentative fwrf model of the classification network intermediary representations.
    _fmaps is a list of grouped feature maps at different resolutions. F maps in total.
    _smsts is a matching resolution stack of batch_t RF model candidates.
    returns a symbolic tensor of receptive field candiate weighted feature maps (bn, features, bv, bt)'''
    _mstfmaps = [T.tensordot(_fm, _smsts[i], [[2,3], [2,3]])  for i,_fm in enumerate(_fmaps)]
    _mst_data = T.concatenate(_mstfmaps, axis=1)
    return _mst_data

def make_normalize_mst_data(_mst_data, nf, nv):
    _sAvg = theano.shared(np.zeros(shape=(1,nf,nv,1), dtype=fpX))
    _sStd = theano.shared(np.zeros(shape=(1,nf,nv,1), dtype=fpX))
    return (_mst_data -  T.patternbroadcast(_sAvg, (True, False, False, False))) / T.patternbroadcast(_sStd, (True, False, False, False)), [_sAvg, _sStd]


########################################################################
###              THE MAIN MODEL FUNCTION                             ###
########################################################################

def model_space(model_specs):
    vm = np.asarray(model_specs[0])
    nt = np.prod([sms.length for sms in model_specs[1]])           
    rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(model_specs[1])]
    xs, ys, ss = np.meshgrid(rx, ry, rs, indexing='ij')    
    return np.concatenate([xs.reshape((nt,1)).astype(dtype=fpX), 
                           ys.reshape((nt,1)).astype(dtype=fpX), 
                           ss.reshape((nt,1)).astype(dtype=fpX)], axis=1)

def model_space_tensor(
        datas, models, feature_envelopes=None, _symbolic_feature_maps=None, fmaps_sizes=None, _symbolic_input_vars=None, 
        nonlinearity=None, zscore=False, mst_avg=None, mst_std=None, epsilon=1e-6, trn_size=None,
        batches=(1,1), view_angle=20., verbose=False, dry_run=False):
    '''
    batches dims are (samples, candidates)

    Feature maps can be provided symbolically, in which case the input will be something else from which the feature maps would be
    calculated and a symbolic variable has to be provided to represent the input. If the feature maps are directly the input, then all 
    these symbols are created automatically.

    This function returns a 4 dimensional model_space tensor, which has dimensions (samples, total number of features, 1, total number of candidates rf).
    The singleton dimension represent the voxels index. However in our case, all voxels share the same candidates rf which is why this dimension is 1.
    '''
    n = len(datas[0])
    bn, bt = batches
    nt = len(models)         
    mx, my, ms = models[np.newaxis,:,0], models[np.newaxis,:,1], models[np.newaxis,:,2]
    nbt = nt // bt
    rbt = nt - nbt * bt
    assert rbt==0, "the candidate batch size must be an exact divisor of the total number of candidates"
    ### CHOOSE THE INPUT VARIABLES
    print ('CREATING SYMBOLS\n')
    if _symbolic_feature_maps is None:
        _fmaps, fmap_sizes = [], []
        for d in datas:
            _fmaps += [T.tensor4(),] 
            fmap_sizes += [d.shape,]
    else:
        _fmaps = _symbolic_feature_maps
        fmap_sizes = fmaps_sizes
        assert fmap_sizes is not None

    if _symbolic_input_vars is None:
        _invars = _fmaps
        for d,fs in zip(datas,fmap_sizes):
            assert d.shape[1:]==fs[1:]
    else:
        _invars = _symbolic_input_vars
    ### CREATE SYMBOLIC EXPRESSIONS AND COMPILE
    _smsts, nf = create_shared_batched_feature_maps_gaussian_weights(fmap_sizes, 1, bt, verbose=verbose)
    _mst_data = make_mst_data(_fmaps, _smsts)  
    if verbose:
        print (">> Storing the full modelspace tensor will require approx %.03fGb of RAM!" % (fpX(n*nf*nt*4) / 1024**3))
        print (">> Will be divided in chunks of %.03fGb of VRAM!\n" % ((fpX(n*nf*bt*4) / 1024**3)))
    print ('COMPILING...')
    sys.stdout.flush()
    comp_t = time.time()
    mst_data_fn  = theano.function(_invars, _mst_data)
    print ('%.2f seconds to compile theano functions' % (time.time()-comp_t))
    ### EVALUATE MODEL SPACE TENSOR
    start_time = time.time()
    print ("\nPrecomputing mst candidate responses...")
    sys.stdout.flush()
    mst_data = np.ndarray(shape=(n,nf,1,nt), dtype=fpX)   
    if dry_run:
        return mst_data, None, None
    for t in tqdm(range(nbt)): ## CANDIDATE BATCH LOOP     
        # set the receptive field weight for this batch of voxelmodel
        set_shared_batched_feature_maps_gaussian_weights(_smsts, mx[:,t*bt:(t+1)*bt], my[:,t*bt:(t+1)*bt], ms[:,t*bt:(t+1)*bt], env=feature_envelopes, size=view_angle)
        for excerpt, size in iterate_slice(0, n, bn):
            args = slice_arraylist(datas, excerpt)  
            mst_data[excerpt,:,:,t*bt:(t+1)*bt] = mst_data_fn(*args)
    full_time = time.time() - start_time
    print ("%d mst candidate responses took %.3fs @ %.3f models/s" % (nt, full_time, fpX(nt)/full_time))
    ### OPTIONAL NONLINEARITY
    if nonlinearity:
        print ("Applying nonlinearity to modelspace tensor...")
        sys.stdout.flush()
        for rr, rl in tqdm(iterate_slice(0, mst_data.shape[3], bt)): 
            mst_data[:,:,:,rr] = nonlinearity(mst_data[:,:,:,rr])
    ### OPTIONAL Z-SCORING
    mst_avg_loc = None
    mst_std_loc = None
    if zscore:
        if trn_size==None:
            trn_size = len(mst_data)
        print ("Z-scoring modelspace tensor...")
        if mst_avg is not None and mst_std is not None:
            print ("Using provided z-scoring values.")
            sys.stdout.flush()
            assert mst_data.shape[1:]==mst_avg.shape[1:], "%s!=%s" % (mst_data.shape[1:], mst_avg.shape[1:])
            assert mst_data.shape[1:]==mst_std.shape[1:], "%s!=%s" % (mst_data.shape[1:], mst_avg.shape[1:])
            mst_avg_loc = mst_avg 
            mst_std_loc = mst_std
            for rr, rl in tqdm(iterate_slice(0, nt, bt)):   
                mst_data[:,:,:,rr] -= mst_avg_loc[:,:,:,rr]
                mst_data[:,:,:,rr] /= mst_std_loc[:,:,:,rr]
                mst_data[:,:,:,rr] = np.nan_to_num(mst_data[:,:,:,rr])        
        else: # calculate the z-score stat the first time around.
            print ("Using self z-scoring values.")
            sys.stdout.flush()
            mst_avg_loc = np.ndarray(shape=(1,)+mst_data.shape[1:], dtype=fpX)
            mst_std_loc = np.ndarray(shape=(1,)+mst_data.shape[1:], dtype=fpX)
            for rr, rl in tqdm(iterate_slice(0, nt, bt)):   
                mst_avg_loc[0,:,:,rr] = np.mean(mst_data[:trn_size,:,:,rr], axis=0, dtype=np.float64).astype(fpX)
                mst_std_loc[0,:,:,rr] =  np.std(mst_data[:trn_size,:,:,rr], axis=0, dtype=np.float64).astype(fpX) + fpX(epsilon)
                mst_data[:,:,:,rr] -= mst_avg_loc[:,:,:,rr]
                mst_data[:,:,:,rr] /= mst_std_loc[:,:,:,rr]
                mst_data[:,:,:,rr] = np.nan_to_num(mst_data[:,:,:,rr])
    ### Free the VRAM
    set_shared_parameters(_smsts)
    return mst_data, mst_avg_loc, mst_std_loc


def learn_params_ridge_regression(mst_data, voxels, lambdas, voxel_batch_size, holdout_size=100, shuffle=True, add_bias=False):
    from theano.tensor.nlinalg import matrix_inverse
    _Xtrn = T.matrix() # [#sample, #feature] ***
    _Ytrn = T.tensor3() # [#ridge, #feature, #feature]
    _Xout = T.matrix() # [#sample, #feature]
    _Vtrn = T.matrix()  # [#sample, #voxel]
    _Vout = T.matrix()  # [#sample, #voxel]
    print ('COMPILING')
    t = time.time()
    ### pre-factor
    _cov = T.dot(_Xtrn.T, _Xtrn)
    _factor = T.stack([matrix_inverse(_cov + T.identity_like(_cov) * fpX(l)) for l in lambdas], axis=0) # [#candidate, #feature, #feature]
    _cofactor =  T.dot(_factor, _Xtrn.T) # [#candidate, #feature, #sample]
    _beta = T.dot(_Ytrn, _Vtrn) # [#candidate, #feature, #voxel]
    _loss = T.sum(T.sqr(_Vout.dimshuffle(0,'x',1) - T.tensordot(_Xout, _beta, axes=[[1],[1]])), axis=0) # [#candidate, #voxel]        
    ### 
    factor_fn = theano.function([_Xtrn], _cofactor)
    score_fn  = theano.function([_Ytrn, _Vtrn, _Xout, _Vout], [_beta, _loss])
    print ('%.2f seconds to compile theano functions'%(time.time()-t))
    # shuffle
    nt,nf,_,nc = mst_data.shape
    _,nv = voxels.shape
    order = np.arange(len(voxels), dtype=int)
    if shuffle:
        np.random.shuffle(order)
    mst_data = mst_data[order].astype(fpX)
    voxels    = voxels[order].astype(fpX)   
    #
    trn_trn_mst_data = mst_data[:-holdout_size]
    out_trn_mst_data = mst_data[-holdout_size:]
    trn_trn_voxel_data = voxels[:-holdout_size]
    out_trn_voxel_data = voxels[-holdout_size:]
    best_w_params = np.zeros(shape=(nv, nf), dtype=fpX)
    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=fpX)], axis=1)
        trn_trn_mst_data = np.concatenate([trn_trn_mst_data, np.ones(shape=(len(trn_trn_mst_data),1,1,nc), dtype=fpX)], axis=1)
        out_trn_mst_data = np.concatenate([out_trn_mst_data, np.ones(shape=(len(out_trn_mst_data),1,1,nc), dtype=fpX)], axis=1)
    best_scores  = np.full(shape=(nv,), fill_value=np.inf, dtype=fpX)
    best_lambdas = np.full(shape=(nv,), fill_value=-1, dtype=np.int)
    best_candidates = np.full(shape=(nv,), fill_value=-1, dtype=np.int)
    start_time = time.time()
    vox_loop_time = 0
    for c in tqdm(range(nc)):
        cof = factor_fn(trn_trn_mst_data[:,:,0,c]) # [#candidate, #feature, #feature]
        vox_start = time.time()
        for rv,lv in iterate_range(0, nv, voxel_batch_size):
            betas, scores = score_fn(cof, trn_trn_voxel_data[:,rv],\
                                     out_trn_mst_data[:,:,0,c], out_trn_voxel_data[:,rv])
            #   [#candidate, #feature, #voxel, ]     [#candidate, #voxel]
            select = np.argmin(scores, axis=0)
            values = np.amin(scores, axis=0)
            imp = values<best_scores[rv]
            if np.sum(imp)>0:
                arv = np.array(rv)[imp]
                best_lambdas[arv] = np.copy(select[imp])
                best_scores[arv] = np.copy(values[imp])
                best_candidates[arv] = np.copy(c)
                best_w_params[arv,:] = np.copy(pnu.select_along_axis(betas[:,:,imp], select[imp], run_axis=2, choice_axis=0).T) 
        vox_loop_time += (time.time() - vox_start)
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    return_params = [best_w_params[:,:nf],]
    if add_bias:
        return_params += [best_w_params[:,-1],]
    print ('-------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / nv))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / nv))
    print ('setup throughput = %fs/candidate' % (inv_time / nc))
    return best_scores, best_lambdas, best_candidates, return_params


def batched_learn_params_ridge_regression(
        _symbolic_feature_maps, fmaps_sizes, _symbolic_input_vars,
        datas, voxel_data, models, lambdas, sample_batch_size, voxel_batch_size, model_batch_size, model_minibatch_size,
        holdout_size, add_bias=True, mst_nonlinearity=None, mst_zscore=True, shuffle=False, view_angle=1., dtype=np.float32):
    trn_size, nv = voxel_data.shape
    nf = np.sum([fs[1] for fs in fmaps_sizes])    
    best_scores = np.full(shape=(nv), fill_value=np.inf, dtype=dtype)
    
    # if I need to add shuffling of the training set, I need to do it here
    # shuffle fmaps and voxel_data ordering, or trn_mst_data in the loop
    order = np.arange(len(voxel_data), dtype=int)
    if shuffle:
        np.random.shuffle(order)
    voxels = voxel_data[order].astype(fpX)   
    
    vindex = np.arange(nv, dtype=int)
    best_lambdas = np.full(shape=(nv,), fill_value=-1, dtype=np.int)
    best_w_params = [np.zeros(shape=(nv, nf), dtype=fpX),]
    if add_bias:
        best_w_params += [np.zeros(shape=(nv,), dtype=fpX),]
    best_rf_params = np.zeros(shape=(nv, 3), dtype=fpX)
    best_avg = np.zeros(shape=(nv, nf), dtype=fpX)
    best_std = np.zeros(shape=(nv, nf), dtype=fpX)
    
    for mr, ml in iterate_range(0,len(models),model_batch_size):
        print ("================================================")
        print ("==> Candidate block %d--%d of %d" % (mr[0], mr[-1], len(models)))
        print ("================================================")
        sys.stdout.flush()
        model_block = models[mr]
        # fmaps take a lot of space. I may be able to build the mst in batch.
        mst_data, mst_avg, mst_std = model_space_tensor(datas, \
            model_block,\
            _symbolic_feature_maps=_symbolic_feature_maps, \
            fmaps_sizes=fmaps_sizes, \
            _symbolic_input_vars=_symbolic_input_vars,\
            nonlinearity=mst_nonlinearity, \
            zscore=mst_zscore, trn_size=trn_size,\
            epsilon=1e-6, batches=(sample_batch_size, model_minibatch_size), \
            view_angle=view_angle, verbose=True, dry_run=False)
        if shuffle:
            mst_data = mst_data[order]
        # parallelized batched ridge regression
        selected_scores, selected_lambdas, selected_candidates, selected_w_params = \
            learn_params_ridge_regression(mst_data, voxels, lambdas, voxel_batch_size, holdout_size, shuffle=False, add_bias=add_bias)   
        
        imp = selected_scores<best_scores # improvement        
        if np.sum(imp)>0:
            arv = vindex[imp]
            # Convert modelspace tensor best candidates into real space and separate voxel models ###
            imp_rf_params, imp_avg, imp_std = real_space_model(\
                selected_candidates[imp], model_block, mst_avg=mst_avg,\
                mst_std=mst_std) 
        
            best_lambdas[arv] = np.copy(selected_lambdas[imp])
            best_scores[arv] = np.copy(selected_scores[imp])
            for k in range(len(best_w_params)):
                best_w_params[k][arv] = np.copy(selected_w_params[k][imp])
            best_rf_params[arv] = np.copy(imp_rf_params)
            best_avg[arv] = np.copy(imp_avg)
            best_std[arv] = np.copy(imp_std)
        
    return best_scores, best_lambdas, best_w_params, best_rf_params, best_avg, best_std

#log_act_func = lambda x: np.log(1+np.abs(x))*np.tanh(np.abs(x)) #np.log(1+np.sqrt(np.abs(x)))
#_log_act_func = lambda _x: T.log(fpX(1)+T.abs_(_x))*T.tanh(T.abs_(_x)) #T.log(fpX(1)+T.sqrt(T.abs_(_x)))
#    
#sample_batch_size = 200
#voxel_batch_size = 500
#model_batch_size = 100
#holdout_size = 1000
#lambdas = np.logspace(3.5,6.5,6)
#best_scores, best_lambdas, best_w_params, best_rf_params, best_avg, best_std = \
#    batched_learn_params_ridge_regression([fm[:trn_size] for fm in fmaps], trn_voxel_data, models, lambdas, sample_batch_size, voxel_batch_size, model_batch_size,
#                          holdout_size, add_bias=True, mst_nonlinearity=log_act_func)


def get_prediction_from_mst(mst_data, mst_rel_models, w_params, batches=(1,1), verbose=False):
    '''
    batches dims are (samples, voxels)

    Arguments:
     mst_data: The modelspace tensor of the validation set.
     voxels: The corresponding expected voxel response for the validation set.
     mst_rel_models: The relative rf model that have been selected through training.
     params: The trained parameter values of the model.
     batches: (#sample, #voxels) per batch
        
    Returns:
     voxel prediction, per voxel corr_coeff
    '''
    nt, nf, _, nc = mst_data.shape
    nv = len(w_params[0])
    bt, bv = batches
    nbv, rbv = nv // bv, nv % bv
    nbt, rbt = nt // bt, nt % bt
    assert len(mst_rel_models)==nv ## voxelmodels interpreted as relative model  
    assert mst_rel_models.dtype==int
    if verbose:
        print ("%d voxel batches of size %d with residual %d" % (nbv, bv, rbv) )
        print ("%d sample batches of size %d with residual %d" % (nbt, bt, rbt) )
        print ('CREATING SYMBOLS\n')
    _mst_data = T.tensor4()
    _fwrf_pred, fwrf_params = make_batched_regression(_mst_data, nf, bv, 1, add_bias=len(w_params)==2)
    ###
    if verbose:
        print ('COMPILING...')
        sys.stdout.flush()
    comp_t = time.time()
    fwrf_pred_fn = theano.function([_mst_data], _fwrf_pred[:,:,0])     
    if verbose:   
        print ('%.2f seconds to compile theano functions' % (time.time()-comp_t))
        sys.stdout.flush()
    predictions = np.zeros(shape=(nt, nv), dtype=fpX)
    ### VOXEL BATCH LOOP
    slice_time = 0
    slice_mst = np.zeros(shape=(nt,nf,bv,1), dtype=fpX)
    slice_params = [np.zeros(shape=(bv,)+p.shape[1:], dtype=fpX) for p in w_params]
    for rv, lv in tqdm(iterate_range(0, nv, bv), disable=not verbose):
        slice_t = time.time()
        slice_mst[:,:,:lv] = np.take(mst_data[:, :, 0, :, np.newaxis], mst_rel_models[rv].astype(int), axis=2).astype(fpX)  
        for i,p in enumerate(w_params):
            slice_params[i][:lv] = p[rv]
        slice_time += time.time()-slice_t
        set_shared_parameters(fwrf_params, slice_params)
        ###
        pred = np.zeros(shape=(nt, bv), dtype=fpX)
        for rb, lb in iterate_range(0, nt, bt):
            pred[rb] = fwrf_pred_fn(slice_mst[rb])
        predictions[:,rv] = pred[:,:lv]
    set_shared_parameters(fwrf_params)     
    if verbose: 
        print ("time spent slicing mst: %.2fs: %.2fs/it " % (slice_time, slice_time*bv/nv))
    sys.stdout.flush()
    return predictions



def real_space_model(mst_rel_models, models, mst_avg=None, mst_std=None):
    '''
    Convert candidate in the model space tensor into real space, per-voxel models.
    '''
    nv = len(mst_rel_models)
    #rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(model_specs[1])] # needed to map rf's back to visual space
    # rel_to_abs_shared_models(mst_rel_models, rx, ry, rs) ### put back the models in absolute coordinate a.k.a model spec for the next iteration
    best_rf_params = models[mst_rel_models]  
    best_mst_data_avg = None
    best_mst_data_std = None
    if mst_avg is not None and mst_std is not None:
        nf = mst_avg.shape[1]
        best_mst_data_avg = np.ndarray(shape=(nv, nf), dtype=fpX)
        best_mst_data_std = np.ndarray(shape=(nv, nf), dtype=fpX)
        for v in range(nv):
            best_mst_data_avg[v,:] = mst_avg[0,:,0,mst_rel_models[v]]
            best_mst_data_std[v,:] = mst_std[0,:,0,mst_rel_models[v]]
    return best_rf_params, best_mst_data_avg, best_mst_data_std


def set_shared_variables(var_dicts, rf_params, w_params, avg=None, std=None, view_angle=1.):
    ## pooling fields
    set_shared_batched_feature_maps_gaussian_weights(var_dicts['fpf_weight'], rf_params[:,0], rf_params[:,1], rf_params[:,2], size=view_angle) 
    ## feature weights
    set_shared_parameters(var_dicts['w_params'], w_params)
    ## normalization
    if avg is not None and 'mst_norm' in var_dicts.keys():
        set_shared_parameters(var_dicts['mst_norm'], [avg.T.astype(fpX)[np.newaxis,:,:,np.newaxis], 
                                                      std.T.astype(fpX)[np.newaxis,:,:,np.newaxis]])


def get_symbolic_prediction(_symbolic_feature_maps, fmaps_sizes, rf_params, w_params, avg=None, std=None, _nonlinearity=None, view_angle=1.0):
    '''
    Unlike the training procedure which is trained by part, this creates a matching theano expression from end-to-end.
    
    Arguments:
    _symbolic_feature_maps: the symbolic feature maps using for training
    fmaps_sizes: the feature maps sizes
    voxelmodels: the absolute receptive field coordinate i.e. a (V,3) array whose entry are (x, y, sigma)
    params: the feature tuning parameters
    (optional) avg, std: the average and standard deviation from z-scoring.
    (optional) nonlinearity: a callable function f(x) which returns a theano expression for an elementwise nonlinearity.
    view_angle (default 20.0): Same as during training. This just fix the scale relative to the values of the voxelmodels.
    
    Returns:
    A symbolic variable representing the prediction of the fwRF model.
    A dictionary of all the shared variables.
    '''
    shared_var = {}
    nf = np.sum([fm[1] for fm in fmaps_sizes])
    nv = rf_params.shape[0]
    assert rf_params.shape[1]==3
    print ('CREATING SYMBOLS\n')
    _smsts,_ = create_shared_batched_feature_maps_gaussian_weights(fmaps_sizes, nv, 1, verbose=True)
    shared_var['fpf_weight'] = _smsts
    if avg is not None:
        if _nonlinearity is not None:
            _nmst, shared_var['mst_norm'] = make_normalize_mst_data(_nonlinearity(make_mst_data(_symbolic_feature_maps, _smsts)), nf, nv)
            _fwrf_pred, shared_var['w_params'] = make_batched_regression(_nmst, nf, nv, 1, add_bias=len(w_params)==2)
        else:
            _nmst, shared_var['mst_norm'] = make_normalize_mst_data(make_mst_data(_symbolic_feature_maps, _smsts), nf, nv)
            _fwrf_pred, shared_var['w_params'] = make_batched_regression(_nmst, nf, nv, 1, add_bias=len(w_params)==2)
    else:
        if _nonlinearity is not None:
            _fwrf_pred, shared_var['w_params'] = make_batched_regression(_nonlinearity(make_mst_data(_symbolic_feature_maps, _smsts)), nf, nv, 1, add_bias=len(w_params)==2)
        else:
            _fwrf_pred, shared_var['w_params'] = make_batched_regression(make_mst_data(_symbolic_feature_maps, _smsts), nf, nv, 1, add_bias=len(w_params)==2)            
    set_shared_variables(shared_var, rf_params, w_params, avg=avg, std=std, view_angle=view_angle)     
    return _fwrf_pred.flatten(ndim=2), shared_var


def get_prediction_from_fmaps(fmaps, rf_params, w_params, avg, std, _nonlinearity, view_angle, sample_batch_size=1000, voxel_batch_size=100, verbose=False):
    nt, nv = len(fmaps[0]), len(rf_params)
    bv = min(voxel_batch_size, nv)
    _fmaps = [T.tensor4() for f in fmaps]
    fmaps_sizes = [fm.shape for fm in fmaps]
    _fwrf, var_dict = get_symbolic_prediction(
        _symbolic_feature_maps=_fmaps, fmaps_sizes=fmaps_sizes,
        rf_params=rf_params[:bv], 
        w_params=[w[:bv] for w in w_params], 
        avg=avg[:bv], std=std[:bv], 
        _nonlinearity=_nonlinearity, 
        view_angle=view_angle)
    fwrf_fn = theano.function(_fmaps, _fwrf)
    ###
    predictions = np.zeros(shape=(nt, nv), dtype=fpX)
    for rv,lv in tqdm(iterate_range(0,nv,bv), disable=not verbose):
        if lv==bv:
            set_shared_variables(var_dict, rf_params[rv], [w[rv] for w in w_params], avg=avg[rv], std=std[rv], view_angle=view_angle)
        else: #fill to match batch
            part_rf_params=np.copy(rf_params[:bv]).astype(fpX)
            part_rf_params[:lv] = rf_params[rv]
            part_w_params=[np.copy(w[:bv]).astype(fpX) for w in w_params]
            for pwp,p in zip(part_w_params, w_params):
                pwp[:lv]=p[rv]
            part_avg=np.copy(avg[:bv])
            part_avg[:lv] = avg[rv]
            part_std=np.copy(std[:bv])
            part_std[:lv] = std[rv]
            set_shared_variables(var_dict, part_rf_params, part_w_params, avg=part_avg, std=part_std, view_angle=view_angle)
            
        pred = np.zeros(shape=(nt, bv), dtype=fpX)
        for rt,lt in iterate_range(0, nt, sample_batch_size):
            args = [f[rt].astype(fpX) for f in fmaps]
            pred[rt] = fwrf_fn(*args)
        predictions[:,rv] = pred[:,:lv]
    return predictions

# THAT'S SOMETHING MORE SPECIFIC

#def get_prediction(\
#        _symbolic_feature_maps, fmaps_sizes, _symbolic_input_vars, datas,\
#        best_rf_params, best_w_params, best_avg, best_std,\
#        _log_act_func, lx, sample_batch_size, voxel_batch_size):
#    ###
#    from src.data_preparation import get_dnn_feature_maps
#    print ('COMPILING')
#    t = time.time()
#    sfmaps_fn = theano.function(_symbolic_input_vars, _symbolic_feature_maps)
#    print ('%.2f seconds to compile theano functions'%(time.time()-t))
#    ###
#    val_fmaps = get_dnn_feature_maps(datas[0], sfmaps_fn, batch_size=sample_batch_size) 
#    val_pred  = get_prediction_from_fmaps(val_fmaps, best_rf_params, best_w_params, best_avg, best_std,\
#                              _log_act_func, lx, sample_batch_size, voxel_batch_size)
#    del val_fmaps
#    return val_pred



################################################################
###                 K-OUT VARIANTS                           ###
################################################################
def kout_learn_params_ridge_regression(mst_data, voxels, val_sample_order, lambdas, voxel_batch_size, val_part_size=1, holdout_size=1, verbose=False, dry_run=False, test_run=False):
    '''
        learn_params_ridge_regression(mst_data, voxels, lambdas, voxel_batch_size, holdout_size=100, shuffle=True, add_bias=False)

        A k-out variant of the fwrf shared_model_training routine.
        batches dims are (samples, voxels, candidates)
    '''
    data_size, nv = voxels.shape
    num_val_part = int(data_size / val_part_size)
    trn_size = data_size - val_part_size

    assert np.modf(float(data_size)/val_part_size)[0]==0.0, "num_val_part (%d) has to be an exact divisor of the set size (%d)" % (num_val_part, data_size)
    print "trn_size = %d (incl. holdout), holdout_size = %d, val_size = %d\n" % (trn_size, holdout_size, val_part_size)
    model = {}
    if test_run:
        tnv = voxel_batch_size
        print "####################################"
        print "### Test run %d of %d voxels ###" % (tnv, nv)
        print "####################################"       
        k, (vs,ls) = 0, (slice(0, val_part_size), val_part_size)
        
        trn_mask = np.ones(data_size, dtype=bool)
        trn_mask[val_sample_order[vs]] = False # leave out the first batch of validation point
            
        trn_mst_data = mst_data[trn_mask]
        val_mst_data = mst_data[~trn_mask]

        trn_voxel_data = voxels[trn_mask, 0:tnv]
        val_voxel_data = voxels[~trn_mask, 0:tnv]
        ### fit this part ###
        best_scores, best_lambdas, best_candidates, best_w_params = learn_params_ridge_regression(\
            trn_mst_data, trn_voxel_data, lambdas, voxel_batch_size, holdout_size, shuffle=True, add_bias=True)
        #val_scores, best_scores, best_epochs, best_candidates, best_w_params = learn_params_stochastic_grad(\
        #    trn_mst_data, trn_voxel_data, w_param_inits, batches=batches,\
        #    holdout_size=holdout_size, lr=lr, l2=l2, num_epochs=num_epochs, output_val_scores=-1, output_val_every=1, verbose=verbose, dry_run=dry_run)
        val_pred = get_prediction_from_mst(val_mst_data, best_candidates, best_w_params, batches=(val_part_size, tnv))

        model[k] = {}
        model[k]['scores']    = best_scores
        model[k]['lambdas']   = best_lambdas
        model[k]['w_params']  = best_w_params
        model[k]['candidates'] = best_candidates
        model[k]['val_mask']  = ~trn_mask
        #####################
        val_cc = np.zeros(shape=(tnv,), dtype=fpX)    
        for v in tqdm(range(tnv)):    
            val_cc[v] = np.corrcoef(val_pred[:,v], val_voxel_data[:,v])[0,1]

        model['n_parts'] = 1       
        model['val_pred'] = val_pred
        model['val_cc'] = val_cc
       
    else:
        # The more parts, the more data each part has to learn the prediction. It's a leave k-out.
        full_val_pred = np.zeros(shape=voxels.shape, dtype=fpX)
        for k,(vs,ls) in enumerate(iterate_slice(0, data_size, val_part_size)):
            print "################################"
            print "###   Resampling block %2d   ###" % k
            print "################################"
            trn_mask = np.ones(data_size, dtype=bool)
            trn_mask[val_sample_order[vs]] = False # leave out the first batch of validation point
            
            trn_mst_data = mst_data[trn_mask]
            val_mst_data = mst_data[~trn_mask]

            trn_voxel_data = voxels[trn_mask]
            val_voxel_data = voxels[~trn_mask]
            ### fit this part ###
            best_scores, best_lambdas, best_candidates, best_w_params = learn_params_ridge_regression(\
                trn_mst_data, trn_voxel_data, lambdas, voxel_batch_size, holdout_size, shuffle=True, add_bias=True)
            #val_scores, best_scores, best_epochs, best_candidates, best_w_params = learn_params_stochastic_grad(\
            #    trn_mst_data, trn_voxel_data, w_param_inits, batches=batches,\
            #    holdout_size=holdout_size, lr=lr, l2=l2, num_epochs=num_epochs, output_val_scores=0, output_val_every=10, verbose=verbose, dry_run=dry_run)
            val_pred = get_prediction_from_mst(val_mst_data, best_candidates, best_w_params, batches=(val_part_size, voxel_batch_size))

            model[k] = {}
            model[k]['scores']    = best_scores
            model[k]['lambdas']   = best_lambdas
            model[k]['w_params']  = best_w_params
            model[k]['candidates'] = best_candidates
            model[k]['val_mask']  = ~trn_mask
            #####################
            full_val_pred[~trn_mask] = val_pred 
        ##
        full_cc = np.zeros(nv)
        for v in range(nv):
            full_cc[v] = np.corrcoef(full_val_pred[:,v], voxels[:,v])[0,1]
        ## global pred and cc
        model['n_parts'] = num_val_part
        model['val_pred'] = full_val_pred
        model['val_cc'] = full_cc
    return model


def kout_get_prediction_from_mst(mst_data, model, batches=(1,1)):
    '''
        A k-out variant of the fwrf get_prediction(...) routine
    '''
    data_size, nv = len(mst_data), len(model[0]['w_params'][0])
    num_val_part = model['n_parts']
    assert np.prod([k in model.keys() for k in range(num_val_part)])>0
    full_val_pred = np.zeros(shape=(data_size, nv), dtype=fpX)
    for k in tqdm(range(num_val_part)):
        #print "################################"
        #print "###   Resampling block %2d   ###" % k
        #print "################################"
        val_mask = model[k]['val_mask']
        best_w_params   = model[k]['w_params']
        best_candidates = model[k]['candidates']  
        val_mst_data = mst_data[val_mask]
        full_val_pred[val_mask] = get_prediction_from_mst(val_mst_data, best_candidates, best_w_params, batches=batches)
    return full_val_pred


def kout_real_space_model(model, models, mst_avg=None, mst_std=None):
    '''
        A k-out variant of the fwrf real_space_model(...) routine
    '''
    num_val_part = model['n_parts']
    assert np.prod([k in model.keys() for k in range(num_val_part)])>0
    for k in range(num_val_part):
        mst_rel_models = model[k]['candidates']

        best_rf_params, best_mst_data_avg, best_mst_data_std = real_space_model(mst_rel_models, models, mst_avg=mst_avg, mst_std=mst_std)
        model[k]['rf_params'] = best_rf_params
        model[k]['norm_avg'] = best_mst_data_avg
        model[k]['norm_std'] = best_mst_data_std
    return model


############## MODEL ANALYSIS BY PARTS FOR k-OUT METHODS ################

def get_model_sample_with_replacement(model):
    n = len(model[0]['val_mask'])
    sl  = n / model['n_parts']
    seq = np.random.randint(16,size=16)
    idx = np.arange(n)
    sam = np.zeros(shape=(n), dtype=int)
    for k,s in enumerate(seq):
        sam[k*sl:(k+1)*sl] = idx[model[s]['val_mask']]
    return sam

def prediction_accuracy(pred, data):
    nv = data.shape[1]
    val_cc = np.zeros(shape=(nv), dtype=data.dtype)
    for v in tqdm(range(nv)):
        val_cc[v] = np.corrcoef(pred[:,v], data[:,v])[0,1]
    return val_cc

def kout_get_partial_predictions_from_mst(model, mst, rlist):
    nt,nf,_,nc = mst.shape
    partition_val_pred = np.zeros(shape=(len(rlist),)+(nt, len(model[0]['w_params'][0])), dtype=fpX)
    for l,r in enumerate(rlist):
        part_model = {}
        part_model['n_parts'] = model['n_parts']
        for k in range(model['n_parts']):
            part_model[k] = {}
            part_model[k]['val_mask'] = model[k]['val_mask']
            part_model[k]['candidates'] = model[k]['candidates']
            ##
            best_w_params = model[k]['w_params']
            partition_params = [np.zeros(p.shape, dtype=fpX) for p in best_w_params]  
            partition_params[0][:, r] = best_w_params[0][:, r]
            part_model[k]['w_params'] = partition_params
        partition_val_pred[l,...] = kout_get_prediction_from_mst(mst, part_model, batches=(nt, 1000))
    return partition_val_pred

def multi_cov(A,B):
    def elementwise_cov(a,b):
        c = np.cov(a,b)
        return np.array([c[0,1],c[0,0],c[1,1]])
    vcov = np.vectorize(elementwise_cov, signature='(n),(n)->(3)')
    return vcov(A.T,B.T)

def relative_covariance(model, partition_val_pred, voxel, n_sample):
    sf = partition_val_pred.shape[0]
    nv = partition_val_pred.shape[2]
    partition_cc = np.ndarray(shape=(n_sample, nv))
    partition_ri = np.ndarray(shape=(n_sample, sf, nv))
    partition_rc = np.ndarray(shape=(n_sample, sf, nv))       
    print "##########################################################"
    print "###           Resampling with replacement              ###"
    print "##########################################################"
    sys.stdout.flush()
    for s in tqdm(range(n_sample)):
        idxs = get_model_sample_with_replacement(model) #idxs = np.arange(len(models[run][0]['val_mask']))
        full_cov_01, full_var_0, full_var_1 = list(multi_cov(model['val_pred'][idxs], voxel[idxs]).T)
        partition_cc[s,:] = full_cov_01 / np.sqrt(full_var_0*full_var_1)
        for l in range(sf):
            part_cov_01, part_var_0, part_var_1 = list(multi_cov(partition_val_pred[l,idxs], voxel[idxs]).T)
            partition_ri[s,l,:] = part_cov_01 / np.sqrt(full_var_0*full_var_1)
            partition_rc[s,l,:] = part_cov_01 / np.sqrt(part_var_0*part_var_1)
    return {'val_cc': partition_cc, 'val_ri': partition_ri, 'val_rc': partition_rc}
