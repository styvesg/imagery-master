import sys
import os
import numpy as np
import pickle
from tqdm import tqdm
import time

import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse, det

import src.fwrf as fwrf
import src.numpy_utility as pnu
fpX  = fwrf.fpX

#####################################################################
####                    UTILITY STUFF
#####################################################################
def log2(v):
    return np.log(v)/np.log(2)

def dots(matrices):
    p = matrices[0]
    for m in matrices[1:]:
        p = np.dot(p, m)
    return p

def averageVariance(r):
    return np.mean(np.var(r, axis=0))

def averageTrace(M):
    return np.mean(np.diag(M))

def BuildMask(partitions, layers):
    mask = np.zeros(shape=(sum(partitions)), dtype=bool)
    for l in layers:
        mask[sum(partitions[:l]):sum(partitions[:l+1])] = True
    return mask

def Unmask(V, mask):
    '''the mask determined the shape of the output and tells where to put the values provided by V'''
    A = np.zeros_like(mask, dtype=V.dtype)
    A[mask] = V
    return A

def UnmaskM(M, mask):
    '''the mask determined the last dimension of the shape of the output and tells where to put the block values provided by M. THe first dimension is provided by M'''
    A = np.zeros(shape=(len(M), len(mask)), dtype=M.dtype)
    A[:,mask] = M
    return A


#####################################################################
####             RECEPTIVE FIELD CONSTRUCTION
#####################################################################
def rulespace(f, start, stop, n):
    y = f(np.arange(n))
    scale = (stop-start) / (y[-1] - y[0])
    return start + (y-y[0]) * scale

def spiderweb(radii):
    deltas = radii[1:]-radii[:-1]
    X,Y = [0.,], [0.,]
    phase = 0
    for d,r in zip(deltas,radii[1:]):
        if d!=0:          
            theta = np.linspace(0, 2*np.pi, np.min([8,int(np.ceil(2*np.pi*r/d))]))
            X += [r*np.cos(t+phase) for t in theta]
            Y += [r*np.sin(t+phase) for t in theta]
            phase += theta[1]/np.e
    return np.array(X),np.array(Y)

class size_ft(object):
    def __init__(self, smin, smax, a, b, va, vb):
        self.smin = smin
        self.smax = smax
        self.slope = a
        self.intp = b
        self.va = va
        self.vb = vb
    def __call__(self, d, n):
        r = np.abs(np.random.normal(0, self.vb +self.va*d, size=(n)))
        return np.clip(d*self.slope + self.intp + r, self.smin, self.smax)

def freq_fn(s):
    return 0.5 / s
    
def ori_fn(n):
    return np.linspace(0, np.pi*(n-1)/n, n)


def spatial_fn(x1, y1, x2, y2, s, k, phase):
    p = (x1-x2)*k[0]+(y1-y2)*k[1]
    d = (np.square(x1-x2)+np.square(y1-y2))/s**2
    return np.exp(-d/2)*np.cos(p+phase)

def normalize(v):
    return v / np.sqrt(np.sum(np.square(v)))



def create_wavelet(grid, xy, orientation, size, freq, phase=0):
    fc, gs = grid[0].size, grid[0].size
    k = np.array([np.cos(orientation)*2*np.pi*freq, np.sin(orientation)*2*np.pi*freq]) #wavenumber vector
    return spatial_fn(grid[0],grid[1],xy[0],-xy[1],size,k,phase).flatten()  
    
def create_stack(grid_out, grid_in, freq_fn, size_fn, n_size, ori_fn, n_ori, complex_cell=False, symmetric=False):
    # each wavelet comes in a pair
    # even value position are cos(), odd position are sin()
    X,Y = grid_in
    filter_list = []
    feature_pattern = []
    ori_range = ori_fn(n_ori)  
    for xy in zip(X.flatten(), Y.flatten()):
        size_range = size_fn(np.sqrt(xy[0]**2 + xy[1]**2), n_size) # size may depend on position
        freq_range = freq_fn(size_range)
        ###
        for o in ori_range:
            for s,f in zip(size_range, freq_range):
                rand_phase = np.random.uniform(0., 2*np.pi, size=1)
                if complex_cell:
                    feature_pattern += [(xy[0], xy[1], s, o, f), (xy[0], xy[1], s, o, f)]
                    filter_list += [create_wavelet(grid_out, xy, o, s, f, phase=rand_phase),\
                                    create_wavelet(grid_out, xy, o, s, f, phase=rand_phase+np.pi/2),]
                    if symmetric:
                        feature_pattern += [(-xy[0], -xy[1], s, o, f), (-xy[0], -xy[1], s, o, f)]
                        filter_list += [create_wavelet(grid_out, (-xy[0], -xy[1]), o, s, f, phase=rand_phase+np.pi),\
                                        create_wavelet(grid_out, (-xy[0], -xy[1]), o, s, f, phase=rand_phase+np.pi*3./2),]
                else:
                    feature_pattern += [(xy[0], xy[1], s, o, f),]
                    filter_list += [create_wavelet(grid_out, xy, o, s, f, phase=rand_phase),]  
                    if symmetric:
                        feature_pattern += [(-xy[0], -xy[1], s, o, f),]
                        filter_list += [create_wavelet(grid_out, (-xy[0], -xy[1]), o, s, f, phase=rand_phase+np.pi),] 
    return np.stack(filter_list, axis=0).T, feature_pattern


def create_filter_stack(base_grid, location_stack, size_fn_stack, ori_per_location=4,\
                        size_sample_per_location=1, complex_cell=True, symmetric=False):
    Us,patterns = [],[]
    for l,s_fn in zip(location_stack, size_fn_stack):
        u, p = create_stack(base_grid, l, freq_fn=freq_fn, size_fn=s_fn, n_size=size_sample_per_location,\
                            ori_fn=ori_fn, n_ori=ori_per_location, complex_cell=complex_cell, symmetric=symmetric)
        Us += [u,]
        patterns += p
    return Us, np.stack(patterns, axis=0)

#####################################################################
####                EXPLICIT GAIN EXPRESSION 
#####################################################################
class Omega_ft(object):
    def __init__(self, A1, U2, A2):
        self.pre = np.dot(A1, U2)
        self.pst = np.dot(A2, U2.T)
    def __call__(self, X=None, backward=False):
        if X is not None:
            if backward:
                return dots([self.pst,X,self.pre])
            else:
                return dots([self.pre,X,self.pst])
        else:
            if backward:
                return np.dot(self.pst,self.pre)
            else:
                return np.dot(self.pre,self.pst)
            
class Gain_ft(object):
    def __init__(self, Omega, kappa):
        self.Omega_fn = Omega
        self.kappa = kappa 
    def __call__(self, X=None, backward=False):
        Echo = self.kappa * self.Omega_fn(X, backward=backward)
        return np.linalg.inv(np.eye(len(Echo)) - Echo)
    
def LoopFactorInv(Us, kappas, alphas):
    return [np.dot(U.T,U) + (k+a)*np.eye(len(U.T)) for U,k,a in zip(Us,kappas,alphas)]

def LoopFactors(Us, kappas, alphas):
    return [np.linalg.inv(np.dot(U.T,U) + (k+a)*np.eye(len(U.T))) for U,k,a in zip(Us,kappas,alphas)]
    
def OmegaFactors(As, Us):
    return [Omega_ft(A,U,Ap) for A,U,Ap in zip(As[:-1], Us[1:], As[1:])]
    
def GainFactors(Omegas, kappas):
    return [Gain_ft(Om, k) for Om,k in zip(Omegas, kappas[:-1])]

def LayerGain(l, As, Us, Gs, kappas, backward=False):
    L = len(Gs)
    print 'layer %d of %d' % (l,L+1)  
    st = 'I'
    if backward:
        gain = np.eye(len(As[0]))
        for k in range(0, l-1):
            st = 'H_%d(%s)' % (k+1,st)
            gain = Gs[k](gain, backward=backward)
        print 'k[%d]%sA[%d]U[%d]' % (l,st,l,l+1)
        return kappas[l-1]*dots([gain, As[l-1], Us[l]])    
    else:
        gain = np.eye(len(As[L]))
        for k in range(l-1, L)[::-1]:
            st = 'G_%d(%s)' % (k+1,st)
            gain = Gs[k](gain, backward=backward)
        print '%sA[%d]U^T[%d]' % (st,l,l)
        return dots([gain, As[l-1], Us[l-1].T])

def TotalGain(As, Us, Gs, kappas, source=0):
    # divide the layers into below and above source.
    below = range(1,source) # the layers below the source.
    above = range(source+1,len(As)+1) # the layers above the source.
    print below, above
    gains_below, gains_above = [], []
    if len(below)>0:
        LG = [LayerGain(l, As, Us, Gs, kappas, backward=True) for l in below[::-1]]
        gains_below = [LG[0],]
        for lg in LG[1:]:
            gains_below += [np.dot(lg, gains_below[-1]),]  
    if len(above)>0:
        LG = [LayerGain(l, As, Us, Gs, kappas) for l in above]
        gains_above = [LG[0],]
        for lg in LG[1:]:
            gains_above += [np.dot(lg, gains_above[-1]),]          
    if source>0:
        return gains_below[::-1]+[np.eye(len(As[source-1])),]+gains_above
    else:
        return gains_above


#####################################################################
####       EXPLICIT JOINT PRECISION MATRIX CONSTRUCTION
#####################################################################
def ZEROS(a,b):
    return np.zeros(shape=(a.shape[0], b.shape[1]))
    
def BuildLine(i, Us, Ls):
    #print "len Us = %d" % len(Us)
    assert len(Us)==len(Ls)-1 # Ls also contain the covariance of the prior.
    head = [ZEROS(Ls[i], Ls[k]) for k in range(0,i-1)]
    #print "len head = %d" % len(head)
    if i==0:
        body = [Ls[i], -np.dot(Ls[i], Us[i]),]
    elif i==len(Ls)-1:
        body = [-np.dot(Us[i-1].T, Ls[i-1]), Ls[i]+np.dot(Us[i-1].T, np.dot(Ls[i-1], Us[i-1])),]
    else:
        body = [-np.dot(Us[i-1].T, Ls[i-1]), Ls[i]+np.dot(Us[i-1].T, np.dot(Ls[i-1], Us[i-1])), -np.dot(Ls[i], Us[i])]
    #print "len body = %d" % len(body)
    tail = [ZEROS(Ls[i], Ls[k]) for k in range(i+2,len(Ls))]
    #print "len tail = %d" % len(tail)
    return head+body+tail

def BuildBlockMatrix(B, verbose=False):
    line, partitions = [], []
    for i,b in enumerate(B):
        if verbose:
            print "line %d" % i, ": ", [bb.shape for bb in b]
        line += [np.concatenate(b, axis=1),]
        partitions += [len(b[0]),]
    return np.concatenate(line, axis=0), partitions 


#####################################################################
####                SYMBOLIC GAIN EXPRESSION 
#####################################################################
def _dots(matrices):
    p = matrices[0]
    for m in matrices[1:]:
        p = T.dot(p, m)
    return p

class _Omega_ft(object):
    def __init__(self, _A1, _U2, _A2):
        self.pre = T.dot(_A1, _U2)
        self.pst = T.dot(_A2, _U2.T)
    def __call__(self, _X=None, backward=False):
        if _X is not None:
            if backward:
                return _dots([self.pst,_X,self.pre])
            else:
                return _dots([self.pre,_X,self.pst])
        else:
            if backward:
                return T.dot(self.pst,self.pre)
            else:
                return T.dot(self.pre,self.pst)
            
class _Gain_ft(object):
    def __init__(self, _Omega, kappa):
        self._Omega_fn = _Omega
        self.kappa = kappa 
    def __call__(self, _X=None, backward=False):
        _Echo = self.kappa * self._Omega_fn(_X, backward=backward)
        return matrix_inverse(T.identity_like(_Echo) - _Echo)
    
def _LoopFactorInv(_Us, kappas, alphas):
    return [T.dot(_U.T,_U) + (k+a)*T.eye(_U.shape[1]) for _U,k,a in zip(_Us,kappas,alphas)]

def _LoopFactors(_Us, kappas, alphas):
    L = []
    for _U,k,a in zip(_Us,kappas,alphas):
        _A = T.dot(_U.T,_U)
        L += [matrix_inverse(_A + T.identity_like(_A)*(k+a)),]
    return L
    
def _OmegaFactors(_As, _Us):
    return [_Omega_ft(_A,_U,_Ap) for _A,_U,_Ap in zip(_As[:-1], _Us[1:], _As[1:])]
    
def _GainFactors(_Omegas, kappas):
    return [_Gain_ft(_Om, k) for _Om,k in zip(_Omegas, kappas[:-1])]

def _LayerGain(l, _As, _Us, _Gs, kappas, backward=False):
    L = len(_Gs)
    if backward:
        gain = T.identity_like(_As[0])
        for k in range(0, l-1):
            gain = _Gs[k](gain, backward=backward)
        return kappas[l-1]*_dots([gain, _As[l-1], _Us[l]])    
    else:
        gain = T.identity_like(_As[L])
        for k in range(l-1, L)[::-1]:
            gain = _Gs[k](gain, backward=backward)
        return _dots([gain, _As[l-1], _Us[l-1].T])

def _TotalGain(_As, _Us, _Gs, kappas, source=0):
    # divide the layers into below and above source.
    below = range(1,source) # the layers below the source.
    above = range(source+1,len(_As)+1) # the layers above the source.
    print below, above
    gains_below, gains_above = [], []
    if len(below)>0:
        LG = [_LayerGain(l, _As, _Us, _Gs, kappas, backward=True) for l in below[::-1]]
        gains_below = [LG[0],]
        for lg in LG[1:]:
            gains_below += [T.dot(lg, gains_below[-1]),]  
    if len(above)>0:
        LG = [_LayerGain(l, _As, _Us, _Gs, kappas) for l in above]
        gains_above = [LG[0],]
        for lg in LG[1:]:
            gains_above += [T.dot(lg, gains_above[-1]),]          
    if source>0:
        return gains_below[::-1]+[T.identity_like(_As[source-1]),]+gains_above
    else:
        return gains_above

def _msr_fn(_x, _y):
    return T.mean(T.sqr(_x - _y), axis=0)

def _cc_fn(_x, _y):
    _mx, _my = T.mean(_x, axis=0), T.mean(_y, axis=0)
    _vx, _vy = T.mean(T.sqr(_x - _mx), axis=0), T.mean(T.sqr(_y - _my), axis=0)
    _cxy = T.mean((_x - _mx)*(_y - _my), axis=0)
    return _cxy / T.sqrt(_vx*_vy + 1e-6) * -1 ### inverse correlation



#####################################################################
####                
#####################################################################

def infer_generative_model_params(
        target_RFs, stim_data, trn_size, tst_size, noise=0.01, num_turns=3,
        gammas = np.logspace(-1., 4., 48), z_score=True,
        dtype=np.float32):
    '''stim_data needs to be de-mean-ed'''
    def explicit_posterior_gain(Us, Ls):
        import scipy.linalg as linalg
        blocks = [BuildLine(l, Us, Ls) for l in range(len(Ls))]
        Lambda, partition_sizes = BuildBlockMatrix(blocks)
        s_mask = BuildMask(partition_sizes, layers=[0,])
        r_Sigma2 = np.linalg.inv(Lambda[~s_mask][:,~s_mask])
        return -np.dot(r_Sigma2, Lambda[~s_mask][:,s_mask]), Lambda[~s_mask][:,~s_mask] # [Lambda_aa]^-1*[Lambda_ab], Bishop, p.87
    def explicit_param_posterior(R, beta):
        import scipy.linalg as linalg
        Us = [np.dot(np.dot(r.T, rp1), np.linalg.inv(np.dot(rp1.T, rp1) + beta * np.eye(rp1.shape[1]))) for r,rp1 in zip(R[:-1], R[1:])]
        Ls = [np.linalg.inv(np.dot((r-np.dot(rp1, u.T)).T, r-np.dot(rp1, u.T))/len(r)) for u,r,rp1 in zip(Us, R[:-1], R[1:])]
        Ls += [np.linalg.inv(np.dot(R[-1].T, R[-1])/len(R[-1])),]
        return [Us, Ls]    
    def score(val, pred):
        sc = np.zeros(shape=(pred.shape[1]))
        for v in range(pred.shape[1]):
            sc[v] = np.corrcoef(val[:,v], pred[:,v])[0,1]
        return sc
    def kl(m0, m1, s0, L1):
        kld = np.trace(np.dot(np.dot(m1-m0, L1), (m1-m0).T)) / len(m1)
        v = np.linalg.eigvalsh(L1)
        kld += np.sum(np.square(s0)*v - np.log(v))
        return -0.5*kld

    L = len(target_RFs)
    R0 = stim_data.reshape((len(stim_data),-1)).astype(dtype)
    Rs = []
    for l in range(L):
        Rv = np.dot(R0, target_RFs[l])
        if z_score:
            Rv -= np.mean(Rv, axis=0, keepdims=True)
            Rv /= np.std(Rv, axis=0, keepdims=True) 
            Rv = np.sqrt(1-noise**2)*Rv + np.random.normal(0,noise,size=Rv.shape)
        else:
            Rv += np.random.normal(0,noise,size=Rv.shape)
        Rs += [Rv.astype(dtype),]
    Rv = np.concatenate(Rs, axis=1)
    R_trn          = [R0[:trn_size-tst_size],]+[r[:trn_size-tst_size] for r in Rs]
    R0_tst, Rv_tst = R0[trn_size-tst_size:trn_size], Rv[trn_size-tst_size:trn_size]
    R0_val, Rv_val = R0[trn_size:], Rv[trn_size:]

    ### write the joint precision matrix
    Us = [np.random.normal(0,1,size=target_RFs[0].shape).astype(float),]
    for a,b in zip(target_RFs[:-1],target_RFs[1:]):
        Us += [np.random.normal(0,1,size=(a.shape[1], b.shape[1])).astype(float),]
    Ls = [np.eye(target_RFs[0].shape[0]).astype(float),]
    for a in target_RFs:
        Ls += [np.eye(a.shape[1]).astype(float),]
    import scipy.linalg as linalg
    # Build the precision matrix of the joint distribution of the generative model.
    blocks = [BuildLine(l, Us, Ls) for l in range(len(Ls))]
    Lambda, partition_sizes = BuildBlockMatrix(blocks, verbose=True)

    ###
    gain, pre = explicit_posterior_gain(Us, Ls)
    pred = np.dot(R0, gain.T) 
    # candidates contain all the layer parameters estimated for all the prior values
    # we need to pick the best combination through coordinate ascent.
    candidates = []
    for k,g in enumerate(gammas):
        print " ===> k = %d, gamma = %f <===" % (k,g)
        sys.stdout.flush() 
        candidates += [explicit_param_posterior(R_trn, g),]  
            
    ### evaluate the loss with a set of parameters, find the best set of parameters and repeat until convergence. This in effect find the best combination of beta
    ### tuned to our feedback parameters.
    cUs, cLs = [np.copy(u) for u in candidates[0][0]], [np.copy(l) for l in candidates[0][1]] # we never need to modify layer L
    bUs, bLs = [np.copy(u) for u in candidates[0][0]], [np.copy(l) for l in candidates[0][1]]
    # print [u.shape for c in cUs]
    # print [l.shape for l in cLs]
    ###
    for epoch in range(num_turns): 
        start_time = time.time()
        for l in range(L): # for each layer
            print "Turn %d, Layer %d" % (epoch, l)
            sys.stdout.flush() 
            best_score, best_gamma = -np.inf,-1
            for k,c in tqdm(enumerate(candidates)): # for each value of gamma, find the best.
                cUs[l] = c[0][l]
                cLs[l] = c[1][l]
                gain, pre = explicit_posterior_gain(cUs, cLs)
                tst_pred = np.dot(R0_tst, gain.T)      
                #tst_score = np.sum(score(Rv_tst, tst_pred)) # kl(Rv_tst, tst_pred, noise, pre)# np.sum(score(Rv_tst, tst_pred))
                tst_score = kl(Rv_tst, tst_pred, noise, pre)
                #print "score = ", tst_score
                if tst_score>best_score:
                    best_score = tst_score
                    bUs[l], bLs[l] = np.copy(cUs[l]), np.copy(cLs[l])          
                    best_gamma = k      
            print "  Layer %d, best gamma @ %d (0--%d) with SCORE = %f" % (l,best_gamma,len(candidates),best_score)         
            cUs[l], cLs[l] = np.copy(bUs[l]), np.copy(bLs[l]) # set current value to the best in for that layer.
        ###
        gain, pre = explicit_posterior_gain(bUs, bLs)
        val_pred = np.dot(R0_val, gain.T) 
        Rcc = score(Rv_val, val_pred)
        print '  End of turn: avg cc = %f' % np.mean(np.nan_to_num(Rcc))
        sys.stdout.flush() 
    ###
    return bUs, bLs, Rv_val, val_pred, Rcc

def calculate_activity_during_vision_under_different_clamping(probes, clamping_data, Us, Ls, clamping_layers=[],\
                                                              complex_cell=False, name='probes', store=None, store_probes=True):
    '''
        Both 'probes' and 'clamping data' need to be de-mean-ed.
        Layer 0, the sensor, is always clamped'''
    import scipy.linalg as linalg
    blocks = [BuildLine(l, Us, Ls) for l in range(len(Ls))]
    Lambda, partition_sizes = BuildBlockMatrix(blocks)

    index = np.arange(len(probes))
    np.random.shuffle(index)
    
    activities = store
    if activities is None:
        activities = {}
    # vision
    print "calculating vision responses"
    sys.stdout.flush()

    s_mask = BuildMask(partition_sizes, layers=[0,])
    l_mask = [BuildMask(partition_sizes, layers=[l,]) for l in range(0,len(partition_sizes))]
    data = probes
    if clamping_data is not None:
        data = clamping_data  
    r0_pcp = data.reshape((-1,partition_sizes[0]))
    r_Sigma = np.linalg.inv(Lambda[~s_mask][:,~s_mask])
    mu_transform = np.dot(r_Sigma, Lambda[~s_mask][:,s_mask]) # [Lambda_aa]^-1*[Lambda_ab], Bishop, p.87
    # expected value of the responses of the internal representations during vision:
    r_pcp = -np.dot(r0_pcp, mu_transform.T) 
    # concatenation of the stimuli and responses during vision:
    sr_pcp = np.concatenate([r0_pcp, r_pcp], axis=1)
    #r_Sigma = linalg.cholesky(r_Sigma2)
    if store_probes:
        activities[name] = {}
        activities[name]['r_mu'] = r_pcp    # mean response of all the linear units.
        activities[name]['r_Sigma'] = r_Sigma # covariance of all the linear units.
    
    # imagery from the following clamped layers:
    if clamping_data is not None:
        r0_pcp = probes.reshape((-1,partition_sizes[0]))
        for i in clamping_layers:
            print "calculating responses to clamping layer %d" % i
            sys.stdout.flush()
            si_mask = BuildMask(partition_sizes, layers=[0,i])
            # concatenation of all the clamped layers (vision to 0, layer=l to vision):
            si_img = np.concatenate([r0_pcp, sr_pcp[:,l_mask[i]]], axis=1)
            ri_Sigma = np.linalg.inv(Lambda[~si_mask][:,~si_mask])
            mui_transform = np.dot(ri_Sigma, Lambda[~si_mask][:,si_mask])
            # expected value of the responses of the internal representations during vision without the clamped layer:
            ri_img = -np.dot(si_img, mui_transform.T) 
            # expected value of the responses of the internal representations during vision:
            r_img = (UnmaskM(ri_img, ~si_mask) + UnmaskM(si_img, si_mask))[:,~s_mask]   ### all voxels, even clamped ones. No stimulus.
      
            ri_mask = si_mask[~s_mask]
            ri_idx = np.arange(len(ri_mask))[~ri_mask]
            ri_Sigma_full = np.zeros(shape=(len(ri_mask),len(ri_mask)), dtype=fpX)
            for j,row in zip(ri_idx, ri_Sigma):
                ri_Sigma_full[j,~ri_mask] = row      
            ####################################
            c = '%s_clamp_%d' % (name,i)
            activities[c] = {}
            activities[c]['r_mu'] = r_img    # mean response of all the linear units.
            activities[c]['r_Sigma'] = ri_Sigma_full # covariance of all the linear units.   
    return activities, partition_sizes


def get_activity_roi(partition_sizes, complex_cell):
    voxel_partitions = np.array(partition_sizes[1:]) / (2 if complex_cell else 1)
    voxel_transitions = [0,]
    for p in voxel_partitions[0:]:
        voxel_transitions += [voxel_transitions[-1]+p] 
    roi_mask = [BuildMask(voxel_partitions, layers=[l,]) for l in range(len(voxel_partitions))]
    voxel_roi = sum([i*m.astype(int) for i,m in zip(range(1,len(roi_mask)+1), roi_mask)])
    roi_map = {i: r'$r_%d$' % i for i in range(1,len(roi_mask)+1)}
    return roi_mask, roi_map, voxel_roi, voxel_transitions, voxel_partitions


def activity(r, complex_cell=True):
    #print "complex_cell = %s" % complex_cell
    if complex_cell:
        even = np.arange(0,r.shape[1],2)
        odd  = np.arange(1,r.shape[1],2)
        return np.sqrt(np.square(r[:,even])+np.square(r[:,odd]))
    else:
        return r 

def get_activity_sample(mu, std, scale=1., complex_cell=True):
    r = mu + np.dot(np.random.normal(loc=0.0,scale=scale,size=mu.shape), std)
    return activity(r, complex_cell)



################################################################################################################
################################################################################################################
################################################################################################################

def sample_encoding_model_fit(
        mst, voxels, trn_size,
        holdout_size, lr, num_epochs, specs, voxel_params, 
        mst_avg, mst_std,
        candidate_batch_size=1000):
    model = {}
    nv = voxels.shape[1]
    trn_mst, val_mst = mst[:trn_size], mst[trn_size:]
    trn_voxels, val_voxels = voxels[:trn_size], voxels[trn_size:]
    ### REGRESSION
    val_scores, best_scores, best_epochs, model['best_candidates'], model['best_w_params']\
            = fwrf.learn_params(trn_mst, trn_voxels, voxel_params, batches=(200, nv//4, candidate_batch_size), \
            holdout_size=holdout_size, lr=lr, l2=0.0, num_epochs=num_epochs, output_val_scores=-1, output_val_every=1, verbose=True, shuffle=True, dry_run=False)
    ### convert to real space
    model['best_rf_params'], model['best_avg'], model['best_std']\
            = fwrf.real_space_model(model['best_candidates'], specs, mst_avg=mst_avg, mst_std=mst_std)
    ###
    model['val_pred'], model['val_cc'] = fwrf.get_prediction(val_mst, val_voxels, model['best_candidates'], model['best_w_params'], batches=(1e4, candidate_batch_size))            
    return model


def aggregate_encoding_model_tuning(
        mst, voxels, trn_size,
        models,
        rlist, # a list of partition mask for whatever we want to evaluate the tuning
        n_samples=1,
        candidate_batch_size=1000):

    fmaps_count = len(rlist)
    nv = voxels[0]['mu'].shape[1]
    val_size = voxels[0]['mu'].shape[0] - trn_size
    val_mst = mst[trn_size:]
 
    partition_val_pred = np.ndarray(shape=(fmaps_count, val_size, nv), dtype=fpX)
    partition_val_cc   = np.ndarray(shape=(fmaps_count, nv), dtype=fpX)

    sample_mu    = np.ndarray(shape=(n_samples,)+voxels[0]['mu'].shape, dtype=fpX)
    sample_Sigma = np.ndarray(shape=(n_samples,)+voxels[0]['Sigma'].shape, dtype=fpX)
    sample_val_pred  = np.ndarray(shape=(n_samples,)+models[0]['val_pred'].shape, dtype=fpX)
    sample_w_params  = np.ndarray(shape=(n_samples,)+models[0]['best_w_params'][0].shape, dtype=fpX)
    sample_rf_params = np.ndarray(shape=(n_samples,)+models[0]['best_rf_params'].shape, dtype=fpX)
    partition_cc = np.ndarray(shape=(n_samples, nv))
    partition_ri = np.ndarray(shape=(n_samples, fmaps_count, nv))
    partition_rc = np.ndarray(shape=(n_samples, fmaps_count, nv))
        
    for n in range(n_samples):
        val_voxels = voxels[n]['mu'][trn_size:]

        sample_mu[n,...] = voxels[n]['mu']
        sample_Sigma[n,...] = voxels[n]['Sigma']
        sample_val_pred[n,...] = models[n]['val_pred']
        sample_w_params[n,...] = models[n]['best_w_params'][0]
        sample_rf_params[n,...] = models[n]['best_rf_params']

        for l,r in enumerate(rlist):
            partition_params = [np.zeros(p.shape, dtype=fpX) for p in models[n]['best_w_params']]
            partition_params[0][:,r] = models[n]['best_w_params'][0][:,r]
            partition_val_pred[l,...], partition_val_cc[l,...] = fwrf.get_prediction(val_mst, val_voxels, models[n]['best_candidates'],\
                                                                                     partition_params, batches=(1e4, candidate_batch_size))
        # calculate covariances full_cov = np.cov(val_pred[:,v],  val_voxel_data[:,v])
        full_cc = models[n]['val_cc']
        for v in range(nv):
            # for all resampling
            partition_cc[n,v] = full_cc[v]
            for l in range(fmaps_count):
                part_cov = np.cov(partition_val_pred[l,:,v], val_voxels[:,v])
                part_cc  = part_cov[0,1] / np.sqrt(part_cov[0,0]*part_cov[1,1])
                partition_ri[n,l,v] = part_cc / full_cc[v]
                partition_rc[n,l,v] = part_cc
                
    return {'mu': sample_mu, 'Sigma': sample_Sigma,
            'rf_params': sample_rf_params, 'w_params': sample_w_params, 'val_pred': sample_val_pred,\
            'val_cc': partition_cc, 'val_ri': partition_ri, 'val_rc': partition_rc}

