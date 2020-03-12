import numpy as np
import h5py
import scipy.io as sio
from glob import glob
import PIL.Image as pim
import pickle


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def sie(x, c=10): 
    '''SparseIntegerEmbedding'''
    y = np.zeros((len(x), c), dtype=np.float32)
    y[np.arange(len(x)), x] = 1
    return y

def standardize(img):
    if img.shape[1]==1:
        return img[:,0]
    else:
        return img.transpose(0,2,3,1)

def load_cifar10(path, npc=3):
    assert npc==1 or npc==3, "Invalid color chanel values. Either 1 or 3."
	 # load the label map
    labelMap = unpickle(path+"batches.meta")['label_names']
	###
    data_list = []
    label_list = []
    trn_size = 0
    for i in range(0,5):
        print "loading data_batch_%d..." % (i+1)
        data = unpickle(path+"data_batch_%d" % (i+1))
        data_list += [(data['data'].reshape((-1,3,32,32)).astype(np.float32)) / 255.,]
        label_list += [sie(np.asarray(data['labels']), len(labelMap)),]
        trn_size += len(data['labels'])

    print "loading test_batch..."
    data = unpickle(path+"test_batch")
    data_list += [(data['data'].reshape((-1,3,32,32)).astype(np.float32)) / 255.,]
    label_list += [sie(np.asarray(data['labels']), len(labelMap)),]
	###
    data = np.concatenate(data_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    if npc==1:
        data = data[:,0:1,...] * 0.299 + data[:,1:2,...] * 0.587 + data[:,2:3,...] * 0.114
    return data, label, labelMap, trn_size




def LoadMNIST(filename):
    with open(filename, 'rb') as f:
        #read header
        header = bytearray(f.read(4))
        datatype = 'int32'
        typecode = header[2]
        
        #for i in header:
        #    print i
        if(typecode==8):
            datatype = '>u1';
        elif(typecode == 12):
            datatype = '>f4'
        elif(typecode == 14):
            datatype = '>f8'    
        print datatype
        
        #read dimensions
        size = np.fromfile(f, '>i4', header[3])
        
        dshape = ()
        count = 1
        for d in range(0,len(size)):
            count *= size[d]
            dshape += (size[d], )
            
        print dshape
        #read data
        data = np.fromfile(f, datatype, count)
        return data.reshape(dshape)   

def load_mnist(path):
    mnist_trn_img = path + 'train-images.idx3-ubyte'
    mnist_trn_lab = path + 'train-labels.idx1-ubyte'
    mnist_val_img = path + 't10k-images.idx3-ubyte'
    mnist_val_lab = path + 't10k-labels.idx1-ubyte'   

    mnist_trn_data = LoadMNIST(mnist_trn_img)
    mnist_val_data = LoadMNIST(mnist_val_img)
    trn_size  = mnist_trn_data.shape[0]
    val_size  = mnist_val_data.shape[0]

    data = np.concatenate((mnist_trn_data, mnist_val_data), axis=0).astype(np.float32)
    data_size = trn_size+val_size

    mnist_min = np.min(data)
    mnist_max = np.max(data)
    data = (data[:,np.newaxis,:,:] - mnist_min) / (mnist_max - mnist_min)

    mnist_trn_label = LoadMNIST(mnist_trn_lab)
    mnist_val_label = LoadMNIST(mnist_val_lab)
    label = np.concatenate((mnist_trn_label, mnist_val_label), axis=0)
    label = sie(label, 10)
    return data, label, None, trn_size




