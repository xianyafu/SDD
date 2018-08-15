import numpy as np, math, operator
import sys
import subprocess
import google.protobuf
import caffe
import google.protobuf.text_format 
from sktensor import dtensor, cp_als
from scipy.sparse import csr_matrix
from paths import *

def load_model(filename):
    model = caffe.proto.caffe_pb2.NetParameter()
    input_file = open(filename, 'r')
    google.protobuf.text_format.Merge(str(input_file.read()), model)
    input_file.close()
    return model
def sparse_weights(T,n,c,h,w):
  a = T.reshape(-1)
  abs_count = np.ones(shape=a.shape)

  s = 0
  for i in range(0,a.size):
    if a[i]==0:
	 s = s+1
  print s,"/",a.size," ",s/a.size

  for i in range(0,a.size):
    if a[i]<0:
       abs_count[i]=-1
       a[i]=abs(a[i])
  locate = np.arange(a.size)
  mask = np.ones(shape=a.shape)
  quick_sort(a,locate,0,locate.size-1)
  sparse_num = int(a.size*0.6)
  for i in range(0,sparse_num):
    a[i]=0
    mask[locate[i]]=0
  new_data = np.empty(shape=a.shape)
  for i in range(0,a.size):
    new_data[locate[i]] = a[i]
  for i in range(0,a.size):
    new_data[i]=new_data[i]*abs_count[i]
  sparse_data = np.reshape(new_data,newshape=T.shape)
  s2 = 0
  for i in range(0,a.size):
    if new_data[i]==0:
	 s2 = s2+1
  print s2,"/",a.size," ",float(s2/a.size)
  return sparse_data

def sparse_weights_2(T,n,c):
  a = T.reshape(-1)
  abs_count = np.ones(shape=a.shape)
  s = 0
  s1 = 0
  for i in range(0,a.size):
    if a[i]==0:
	 s = s+1
    if a[i]>-0.4 and a[i]<0.4:
	 s1 = s1+1
  print s,"/",a.size," ",s/a.size

  for i in range(0,a.size):
    if a[i]<0:
       abs_count[i]=-1
       a[i]=abs(a[i])
  locate = np.arange(a.size)
  mask = np.ones(shape=a.shape)
  quick_sort(a,locate,0,locate.size-1)
  sparse_num = int(a.size*0.5)
  for i in range(0,sparse_num):
    a[i]=0
    mask[locate[i]]=0
  new_data = np.empty(shape=a.shape)
  for i in range(0,a.size):
    new_data[locate[i]] = a[i]
  for i in range(0,a.size):
    new_data[i]=new_data[i]*abs_count[i]
  sparse_data = np.reshape(new_data,newshape=T.shape)
  s2 = 0
  for i in range(0,a.size):
    if new_data[i]==0:
	 s2 = s2+1
  print s2,"/",a.size," ",float(s2/a.size)
  return sparse_data
 
def quick_sort(array,locate, l, r):
    if l < r:
        q = partition(array,locate, l, r)
        quick_sort(array, locate, l, q - 1)
        quick_sort(array, locate, q + 1, r)
 
def partition(array, locate, l, r):
    x = array[r]
    i = l - 1
    for j in range(l, r):
        if array[j] <= x:
            i += 1
            array[i], array[j] = array[j], array[i]
            locate[i], locate[j] = locate[j], locate[i]
    array[i + 1], array[r] = array[r], array[i+1]
    locate[i+1], locate[r] = locate[r], locate[i+1]
    return i + 1

def sparse_model():
  lenet_path='/home/fuxianya/sdd/alexnet/alexnet'
  net = caffe.Classifier(lenet_path + '_deploy.prototxt', lenet_path + '.caffemodel')
  fast_net = caffe.Classifier(lenet_path + '_deploy.prototxt', lenet_path + '.caffemodel')
  model = load_model(lenet_path+'.prototxt')
  layer_num = len(net.layers)
  print(len(net.layers))
  for ind in range(layer_num+1):
     if model.layer[ind].type in ['Convolution']:
	i = ind-1
	print(model.layer[ind].name)
	print("sparse conv")
        weights = net.layers[i].blobs[0].data
	T = dtensor(weights)
	print(weights.shape)
	W = sparse_weights(T,
	                   weights.shape[0],
			   weights.shape[1],
			   weights.shape[2],
			   weights.shape[3])
	np.copyto(fast_net.layers[i].blobs[0].data,W)
     if model.layer[ind].type in ['InnerProduct0']:
	print(ind)
	i = ind-1
	print("sparse inner")
	print(model.layer[ind].name)
        weights = net.layers[i].blobs[0].data
	T = dtensor(weights)
	print(weights.shape)
	W = sparse_weights_2(T,
	                   weights.shape[0],
			   weights.shape[1])
	np.copyto(fast_net.layers[i].blobs[0].data,W)

  fast_net.save(lenet_path+'_sparse.caffemodel')

sparse_model()
