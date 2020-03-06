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

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm 
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import FormatStrFormatter

from analysis import _and,_or
from analysis import pretty_accuracy_fit


### GLOBALS 
mode_colors = {'pcp': '#2b43ac', 'img': '#fd842a'}
mode_markers = {}
roi_colors  = []
roi_markers = ['o', 's', '<', 'D', 'v', '*', 'x']


def display_candidate_loss(scores, nx, ny, ns):
    dis_y = ns // 3 if ns%3==0 else ns//3+1
    s = scores.reshape((nx, ny, ns)).transpose((1,0,2))[::-1,:,:] ## The transpose and flip is just so that the candidate 
    #coordinate maatch the normal cartesian coordinate of the rf position when viewed through imshow.
    idxs = np.unravel_index(np.argmin(s), (nx,ny,ns))
    best = plt.Circle((idxs[1], idxs[0]), 0.5, color='r', fill=False, lw=2)
    
    fig = plt.figure(figsize=(15, 5*dis_y))
    smin = np.min(s)
    smax = np.max(s)
    # print "score range = (%f, %f)" % (smin, smax)
    for i in range(ns):
        plt.subplot(dis_y, 3, i+1)
        plt.imshow(s[:,:,i], interpolation='None', cmap='jet')
        plt.title('sigma canditate = %d' % i)
        plt.clim(smin, smax)
        plt.grid(False)
        if(idxs[2]==i):
            ax = plt.gca()
            ax.add_artist(best)
    return fig


from scipy.optimize import curve_fit
def tuning_func(omega, a, b, sigma, omega0):
    return b+a*np.exp(-np.square(np.log(omega)-np.log(omega0))/2/sigma**2)
###

def set_detailed_roi_layout():
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(16, 16))
    plt.subplots_adjust(left=0.12, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
    ax_im   = plt.subplot2grid((3, 3), (1, 1), colspan=2)
    ax_ips  = plt.subplot2grid((3, 3), (0, 2))
    ax_v3ab = plt.subplot2grid((3, 3), (0, 1))
    ax_v3 = plt.subplot2grid((3, 3), (0, 0))
    ax_v1 = plt.subplot2grid((3, 3), (1, 0))
    ax_v2 = plt.subplot2grid((3, 3), (2, 0))
    ax_v4 = plt.subplot2grid((3, 3), (2, 1))
    ax_lo = plt.subplot2grid((3, 3), (2, 2))
    roi_ax = [ax_ips, ax_v3ab, ax_v3, ax_v1, ax_v2, ax_v4, ax_lo]
    roi_xmask = [0, 0, 0, 0, 1, 1, 1]
    roi_ymask = [0, 0, 1, 1, 1, 0, 0]
    ax_im.axis('off')
    ax_im.grid('off')
    return fig, roi_ax, ax_im, [roi_xmask, roi_ymask]

def set_flat_roi_layout():
    fig, ax = plt.subplots(nrows=1, ncols=7, sharex=True, sharey=True, figsize=(42, 6))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    ax_v1   = plt.subplot2grid((1, 7), (0, 0))
    ax_v2   = plt.subplot2grid((1, 7), (0, 1))
    ax_v3   = plt.subplot2grid((1, 7), (0, 2))    
    ax_v3ab = plt.subplot2grid((1, 7), (0, 3))
    ax_v4   = plt.subplot2grid((1, 7), (0, 4))
    ax_lo   = plt.subplot2grid((1, 7), (0, 5))
    ax_ips  = plt.subplot2grid((1, 7), (0, 6))      
    roi_ax = [ax_v1, ax_v2, ax_v3, ax_v3ab, ax_v4, ax_lo, ax_ips]
    roi_xmask = [1, 1, 1, 1, 1, 1, 1]
    roi_ymask = [1, 0, 0, 0, 0, 0, 0]
    return fig, roi_ax, None, [roi_xmask, roi_ymask]


def plot_pretty_compare(x, y, threshold, xlim, ylim, x_color='g', y_color='r', cmap='Blues'):
    from matplotlib.pyplot import cm 
    from matplotlib.ticker import FormatStrFormatter
    cmap = cm.get_cmap(cmap)
    g = sns.JointGrid(x, y, size=8, xlim=xlim, ylim=ylim)
    mask = np.logical_or(x>threshold, y>threshold) #np.where(Xt[1]>threshold)[0]
    _=g.plot_joint(plt.hexbin, bins='log', gridsize=30, cmap=cmap, extent=xlim+ylim)
    ax1=g.ax_marg_x.hist(x,log=True, color=x_color, bins=30, range=xlim) #distplot(color=".5",kde=False) #hist_kws={'log':True}
    ax2=g.ax_marg_y.hist(y,log=True, color=y_color, bins=30, orientation='horizontal', range=ylim)
    
    g.ax_marg_x.get_yaxis().reset_ticks()
    g.ax_marg_x.get_yaxis().set_ticks([1e0, 1e2, 1e4])
    g.ax_marg_x.get_yaxis().set_ticklabels(['$10^0$', '$10^2$', '$10^4$'])
    g.ax_marg_x.set_ylabel('Count', labelpad=10)
    g.ax_marg_x.get_yaxis().grid(True)
    #g.ax_marg_x.get_yaxis().set_major_formatter(FormatStrFormatter('%d'))
    g.ax_marg_y.get_xaxis().reset_ticks()
    g.ax_marg_y.get_xaxis().set_ticks([1e0, 1e2, 1e4])
    g.ax_marg_y.get_xaxis().set_ticklabels(['', '', ''])
    g.ax_marg_y.get_xaxis().grid(True)
    
    mm = [min(xlim[0],ylim[0]), max(xlim[1], ylim[1])]
    g.ax_joint.plot(mm, mm, '--k', lw=2)
    g.ax_joint.plot([mm[0], threshold], [threshold, threshold], '-r', lw=2)
    g.ax_joint.plot([threshold, threshold], [mm[0], threshold], '-r', lw=2)
    
    g.ax_joint.plot([threshold, mm[1]], [threshold, threshold], '--r', lw=2)
    g.ax_joint.plot([threshold, threshold], [threshold, mm[1]], '--r', lw=2)
    return g

def venn_voxel_masks(pic_pcp_mask, pic_img_mask, cue_pcp_mask, cue_img_mask):
    cue_mask = _and(_and(cue_pcp_mask, cue_img_mask), ~_or(pic_pcp_mask, pic_img_mask)) # purple
    straggler_mask = _and(_or(cue_pcp_mask, cue_img_mask), ~_or(cue_mask, _or(pic_pcp_mask, pic_img_mask))) # magenta
    hybrid_mask  = _and(_or(cue_pcp_mask, cue_img_mask), _or(pic_pcp_mask, pic_img_mask)) # red
    pic_mask = _and(_and(pic_pcp_mask, pic_img_mask), ~hybrid_mask) #green
    pcp_xcl_mask = _and(_and(pic_pcp_mask, ~pic_mask), ~hybrid_mask) #blue
    img_xcl_mask = _and(_and(pic_img_mask, ~pic_mask), ~hybrid_mask) #yellow
    none_mask = ~_or(_or(pic_pcp_mask, pic_img_mask), _or(cue_pcp_mask, cue_img_mask)) #gray    
    return pcp_xcl_mask, img_xcl_mask, pic_mask, cue_mask, hybrid_mask, straggler_mask, none_mask

def venn_voxel_map(pic_pcp_mask, pic_img_mask, cue_pcp_mask, cue_img_mask):
    pcp_xcl_mask, img_xcl_mask, pic_mask, cue_mask, hybrid_mask, straggler_mask, none_mask = venn_voxel_masks(pic_pcp_mask, pic_img_mask, cue_pcp_mask, cue_img_mask)
    assert np.sum(cue_mask)+np.sum(straggler_mask)+np.sum(hybrid_mask)\
        +np.sum(pic_mask)+np.sum(pcp_xcl_mask)+np.sum(img_xcl_mask)+np.sum(none_mask)==len(cue_mask), "mask coverage error"
    return pcp_xcl_mask.astype(int) * 1 + img_xcl_mask.astype(int) * 2 + pic_mask.astype(int) * 3 \
        + cue_mask.astype(int) * 4 + hybrid_mask.astype(int) * 5 + straggler_mask.astype(int) * 6
    
def plot_venn_voxels(ax, X, Y, roi_mask, pic_pcp_mask, pic_img_mask, cue_pcp_mask, cue_img_mask, threshold, xlim, ylim):
    pcp_xcl_mask, img_xcl_mask, pic_mask, cue_mask, hybrid_mask, straggler_mask, none_mask = venn_voxel_masks(pic_pcp_mask, pic_img_mask, cue_pcp_mask, cue_img_mask)
    roi_cue_mask = np.logical_and(roi_mask, cue_mask)
    roi_straggler_mask = np.logical_and(roi_mask, straggler_mask)
    roi_hybrid_mask = np.logical_and(roi_mask, hybrid_mask)
    roi_none_mask = np.logical_and(roi_mask, none_mask)
    roi_pic_mask = np.logical_and(roi_mask, pic_mask)                     
    roi_pcp_xcl_mask = np.logical_and(roi_mask, pcp_xcl_mask)
    roi_img_xcl_mask = np.logical_and(roi_mask, img_xcl_mask)
    
    ax.scatter(X[roi_none_mask], Y[roi_none_mask], color='#c4c4c4', alpha=0.5)                
    ax.scatter(X[roi_img_xcl_mask], Y[roi_img_xcl_mask], color='#ffe700', alpha=0.5)
    ax.scatter(X[roi_pcp_xcl_mask], Y[roi_pcp_xcl_mask], color='#003fd8', alpha=0.5)            
    ax.scatter(X[roi_straggler_mask], Y[roi_straggler_mask], color='#ff70e3', alpha=0.5)
    ax.scatter(X[roi_cue_mask], Y[roi_cue_mask], color='#7d00d8', alpha=0.5)   
    ax.scatter(X[roi_hybrid_mask], Y[roi_hybrid_mask], color='#d80000', alpha=0.5)
    ax.scatter(X[roi_pic_mask], Y[roi_pic_mask], color='#169d00', alpha=0.5)
    
    mm = [min(xlim[0],ylim[0]), max(xlim[1], ylim[1])]
    ax.plot(mm, mm, '--k', lw=2)
    ax.plot([mm[0], threshold], [threshold, threshold], '-r', lw=2)
    ax.plot([threshold, threshold], [mm[0], threshold], '-r', lw=2)
    
    ax.plot([threshold, mm[1]], [threshold, threshold], '--r', lw=2)
    ax.plot([threshold, threshold], [threshold, mm[1]], '--r', lw=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def pretty_accuracy_plot(X, Y, mask, good, th, xlim, ylim): 
    a = pretty_accuracy_fit(X[good], Y[good])
    ax = plt.gca()
    ax.hexbin(X[mask], Y[mask], bins='log', gridsize=30, cmap='Greens', extent=xlim+ylim)#
    mm = [min(xlim[0],ylim[0]), max(xlim[1], ylim[1])]
    ax.plot(mm, mm, '--k', lw=2)
    ax.plot([mm[0], th], [th, th], '-r', lw=2)
    ax.plot([th, th], [mm[0], th], '-r', lw=2)
    ax.plot([th, mm[1]], [th, th], '--r', lw=2)
    ax.plot([th, th], [th, mm[1]], '--r', lw=2)
    x = np.linspace(xlim[0], xlim[1], 10)            
    ax.plot(x, a*x, color='r', lw=4)   


def plot_pretty_scatter(ax, X, Y, xlim, ylim):
    mX, mY = np.mean(X), np.mean(Y) 
    sns.kdeplot(X, Y, bins='log', gridsize=30, n_levels=10, shade=True, shade_lowest=False, cmap='Greens', extent=xlim+ylim, ax=ax)
    ax.scatter(X, Y, color='b', marker='.', alpha=0.5)
    ax.scatter(x=[mX,], y=[mY,], marker='o', color='r', lw=6)
    cc = (mX+mY)/2
    ax.plot([cc, mX], [cc, mY], color='r', lw=4)
    ax.plot(xlim, xlim, '--k')  
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return mY-mX




def plot_rf_as_circles(rfs, smin, smax):
    ''' Plots every receptive field as a circle whose center is the center position of the gaussian receptive field and with
        radius corresponding to one standard deviation of that gaussian. 
    '''
    cNorm  = colors.Normalize(vmin=smin, vmax=smax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('jet') )
    # print scalarMap.get_clim()
    for rf in rfs:
        colorVal = scalarMap.to_rgba(rf[2])
        c = plt.Circle((rf[0], rf[1]), rf[2], color=colorVal, fill=True, alpha=0.1, lw=1.)
        plt.gca().add_artist(c)
        c = plt.Circle((rf[0], rf[1]), rf[2], color=colorVal, fill=False, alpha=0.5, lw=2.)
        plt.gca().add_artist(c)
        #plt.scatter([rf[0],], [rf[1],], color=colorVal, marker='o')
    plt.xlim([-15,15])
    plt.ylim([-15,15])
    plt.xlabel('x (degree)')
    plt.ylabel('y (degree)', labelpad=0)
    plt.gca().set_aspect('equal')

def plot_joint_cc_and_cue_and_picture_rf(
        pcp_models, pp_val_cc, cp_val_cc,
        img_models, pi_val_cc, ci_val_cc,
        xlim=[-1,1], ylim=[-1,1], view_angle=1., threshold=0.2, frac_prune=1.):
    ''' A special joint plot of CC with the estimated average receptive field displayed for the voxel responding to the cue and 
        the receptive fields of the voxels responding to the picture.
    '''
    th = threshold
    lx = view_angle
    n_parts = pcp_models['n_parts']
    pcp_rf_params = np.zeros(shape=(n_parts,)+ pcp_models[0]['rf_params'].shape)
    img_rf_params = np.zeros(shape=(n_parts,)+ img_models[0]['rf_params'].shape)
    for k in range(n_parts):
        pcp_rf_params[k,...] = pcp_models[k]['rf_params']
        img_rf_params[k,...] = img_models[k]['rf_params']

    pcp_rf_params_avg = np.mean(pcp_rf_params, axis=0)
    img_rf_params_avg = np.mean(img_rf_params, axis=0)

    pcp_rf_ecc, pcp_prf_size = np.sqrt(np.square(pcp_rf_params_avg[:,0]) + np.square(pcp_rf_params_avg[:,1])), pcp_rf_params_avg[:,2]
    img_rf_ecc, img_prf_size = np.sqrt(np.square(img_rf_params_avg[:,0]) + np.square(img_rf_params_avg[:,1])), img_rf_params_avg[:,2]
    ###
    from scipy.stats import rv_discrete
    _pruning_dist = lambda x: rv_discrete(name='custm', values=([0,1], [x, 1-x]))
    ###
    plt.subplot(2,3,1)
    plt.hexbin(pp_val_cc, cp_val_cc,  bins='log', gridsize=30, cmap='Blues', extent=xlim+ylim)
    plt.axhline(y=th, color='k')
    plt.axvline(x=th, color='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('picture reponses during vision')
    plt.ylabel('cue reponses during vision')
    plt.text(-0.15, 0.6, 'Cue only')
    plt.text(0.6, -0.15, 'Pic only')
    plt.text(0.6, 0.6, 'Hybrid')
    ###
    plt.subplot(2,3,2)
    pcp_cue_rf = pcp_rf_params_avg[np.logical_and(cp_val_cc>th, pp_val_cc<th)]
    mask = _pruning_dist(frac_prune).rvs(size=len(pcp_cue_rf)).astype(bool)
    plot_rf_as_circles(pcp_cue_rf[mask], 0, 8)
    plt.plot([-lx/2,lx/2,lx/2,-lx/2,-lx/2], [lx/2,lx/2,-lx/2,-lx/2, lx/2], 'r')
    ###
    plt.subplot(2,3,3)
    pcp_pic_rf = pcp_rf_params_avg[np.logical_and(pp_val_cc>th, cp_val_cc<th)]
    mask = _pruning_dist(frac_prune).rvs(size=len(pcp_pic_rf)).astype(bool)
    plot_rf_as_circles(pcp_pic_rf[mask], 0, 8)
    plt.plot([-lx/2,lx/2,lx/2,-lx/2,-lx/2], [lx/2,lx/2,-lx/2,-lx/2, lx/2], 'r')
    ########################################################################
    plt.subplot(2,3,4)
    plt.hexbin(pi_val_cc, ci_val_cc,  bins='log', gridsize=30, cmap='Blues', extent=xlim+ylim)
    plt.axhline(y=th, color='k')
    plt.axvline(x=th, color='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('picture reponses during imagery')
    plt.ylabel('cue reponses during imagery')
    ###
    plt.subplot(2,3,5)
    img_cue_rf = img_rf_params_avg[np.logical_and(ci_val_cc>th, pi_val_cc<th)]
    mask = _pruning_dist(frac_prune).rvs(size=len(img_cue_rf)).astype(bool)
    plot_rf_as_circles(img_cue_rf[mask], 0, 8)
    plt.plot([-lx/2,lx/2,lx/2,-lx/2,-lx/2], [lx/2,lx/2,-lx/2,-lx/2, lx/2], 'r')
    ###
    plt.subplot(2,3,6)
    img_pic_rf = img_rf_params_avg[np.logical_and(pi_val_cc>th, ci_val_cc<th)]
    mask = _pruning_dist(frac_prune).rvs(size=len(img_pic_rf)).astype(bool)
    plot_rf_as_circles(img_pic_rf[mask], 0, 8)
    plt.plot([-lx/2,lx/2,lx/2,-lx/2,-lx/2], [lx/2,lx/2,-lx/2,-lx/2, lx/2], 'r')


def plot_sorted_validation_accuracy(tuning_dicts, conditions, cc_th):
    for i,m in enumerate(conditions):
        avg = tuning_dicts[m]['val_cc_sorted']
        std = tuning_dicts[m]['cc_std_sorted']
        p=plt.plot(np.arange(len(avg))+1, avg, color=mode_colors[m])
        plt.fill_between(np.arange(len(avg))+1, avg-std,  avg+std, color=mode_colors[m], alpha=0.5)
        plt.xlabel('index')
        plt.ylabel('prediction accuracy')
        plt.xscale('log')
        plt.ylim([-.25,.85])
        plt.gca().axhline(y=cc_th, color='gray', linestyle='--', lw=2)


def plot_voxel_tuning_samples(support, tuning_dicts, conditions, voxel_indices):
    for k,v in enumerate(voxel_indices):
        plt.subplot(1,len(voxel_indices),k+1)
        for i,m in enumerate(conditions):
            _=plt.plot(support, tuning_dicts[m]['sample_fi'][:,:,v].T, color=mode_colors[m], alpha=0.25)
        plt.xscale('log')
        plt.xlim([support[0]*0.95, support[-1]*1.05])
        plt.title('voxel #%s' % v)


def plot_roi_tuning_curves(support, tuning_dicts, conditions, group_names):
    gn = group_names
    x  = np.linspace(support[0]*0.95, support[-1]*1.05, 100)
    plt.subplots_adjust(left=0.05, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
    for k,mask in enumerate(['all', 'pic']):
        for roi,name in enumerate(gn):
            plt.subplot(2,len(gn),k*len(gn)+roi+1)   
            for i,m in enumerate(conditions):
                avg = tuning_dicts[m]['Fi_%s' % mask][:,roi] / support
		avg = avg / np.sum(avg)
                std = tuning_dicts[m]['dFi_%s' % mask][:,roi] / (support * np.sum(avg))
                plt.errorbar(support, avg, std, linestyle='None', color=mode_colors[m], marker='o', markersize=10) 
                ### curve fit
                popt, pcov = curve_fit(tuning_func, support, avg, p0=[.1, 0., 10., 4.], sigma=std, absolute_sigma=True)
                plt.plot(x, tuning_func(x, *popt), color=mode_colors[m], lw=6)  
            plt.xscale('log')
            plt.xlim([support[0]*0.95, support[-1]*1.05])
            plt.ylim([0., 0.18])
            if k==0:
                plt.title(gn[roi])
    plt.gcf().text(0.5, 0.01, r'$\omega$', ha='center', fontsize=mpl.rcParams['axes.titlesize'])
    plt.gcf().text(0.01, 0.70, 'all', va='center', rotation='vertical', fontsize=mpl.rcParams['axes.titlesize']) 
    plt.gcf().text(0.01, 0.30, 'pic', va='center', rotation='vertical', fontsize=mpl.rcParams['axes.titlesize']) 

