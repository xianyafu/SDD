import caffe
from sktensor import dtensor, cp_als
import numpy as np
if __name__ == "__main__":
  root = '/home/fuxianya/sdd/lenet/'
  caffe.set_mode_cpu
  net = caffe.Net(root + 'lenet.prototxt',\
        "/home/fuxianya/sdd/lenet/lenet.caffemodel",caffe.TEST) #root + '_iter_64000.caffemodel',caffe.TEST)
  weights = net.params['conv2'][0].data
  T = dtensor(weights)
  l01=0
  l02=0
  l03=0
  l1=0
  l2=0
  l3=0
  l4=0
  l5=0
  l6=0
  l7=0
  l8=0
  l9=0
  lo=0
  for i in range(weights.shape[0]):
      for j in range(weights.shape[1]):
	  for k in range(weights.shape[2]):
	      for m in range(weights.shape[3]):
                if T[i,j,k,m]==0:
			lo=lo+1
  print(lo)	

  for i in range(weights.shape[0]):
      for j in range(weights.shape[1]):
	  for k in range(weights.shape[2]):
	      for m in range(weights.shape[3]):
		  if abs(T[i,j,k,m]) <pow(2,-20):
		     l01 = l01 + 1
		  if abs(T[i,j,k,m]) <pow(2,-18) and abs(T[i,j,k,m]) >= pow(2,-20):
		     l02 = l02 + 1
		  if abs(T[i,j,k,m]) <pow(2,-16) and abs(T[i,j,k,m]) >= pow(2,-18):
		     l03 = l03 + 1
		  if abs(T[i,j,k,m]) <pow(2,-14) and abs(T[i,j,k,m]) >= pow(2,-16):
		     l2 = l2 + 1
		  if abs(T[i,j,k,m]) <pow(2,-10) and abs(T[i,j,k,m]) >= pow(2,-12):
		     l3 = l3 + 1
		  if abs(T[i,j,k,m]) <pow(2,-8) and abs(T[i,j,k,m]) >= pow(2,-10):
		     l4 = l4 + 1
		  if abs(T[i,j,k,m]) <pow(2,-6) and abs(T[i,j,k,m]) >= pow(2,-8):
		     l5 = l5 + 1
		  if abs(T[i,j,k,m]) <pow(2,-4) and abs(T[i,j,k,m]) >= pow(2,-6):
		     l6 = l6 + 1
		  if abs(T[i,j,k,m]) <pow(2,-2) and abs(T[i,j,k,m]) >= pow(2,-4):
		     l7 = l7 + 1 
		  if abs(T[i,j,k,m]) <pow(2,0) and abs(T[i,j,k,m]) >= pow(2,-2):
		     l8 = l8 + 1
		  if abs(T[i,j,k,m]) >pow(2,0):
	             l9 = l9 + 1
  print(weights.shape[0]*weights.shape[1]*weights.shape[2]*weights.shape[3],' ',l01," ", l02," ",l03," ",l2," ",l3," ",l4," ",l5," ",l6," ",l7," ",l8," ",l9)
  lo=0
  for i in range(weights.shape[0]):
      for j in range(weights.shape[1]):
	  for k in range(weights.shape[2]):
	      for m in range(weights.shape[3]):
                if T[i,j,k,m]==0:
			lo=lo+1
  print(lo)	
  #np.copyto(net.params['conv1'][0].data,T)
