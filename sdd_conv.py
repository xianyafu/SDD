import numpy as np, math, operator
import sys
import subprocess
import google.protobuf
import caffe
import google.protobuf.text_format 
from sktensor import dtensor, cp_als
from scipy.sparse import csr_matrix
from paths import *


def sdd(A,kmax=100,alphamin=0.01,lmax=100,rhomin=0,yinit=1):

    try: 
        'A'
    except NameError:
        print 'Incorrect number of inputs.'
    
    if 'rhomin' in locals():
        rhomin = math.pow(rhomin,2)
    idx = 0             # only used for yinit = 1 (python is zero-based contrary to matlab)
    
    # Initialization    
    
    [m,n] = A.shape         # size of A
    rho = math.pow(np.linalg.norm(A,'fro'),2)   # squared residual norm
    
    iitssav = np.zeros((kmax))
    xsav = np.zeros((m,kmax))
    xsav = np.asarray(xsav)
    ysav = np.zeros((n,kmax))
    ysav = np.asarray(ysav)
    dsav = np.zeros((kmax,1))
    itssav = np.zeros((kmax))    
    rhosav = np.zeros((kmax))
    A = np.asarray(A)
    betabar = 0    
    
    # Outer loop
    
    for k in range(0,kmax):
        print "k: ", k
        # Initialize y for inner loop
        
        if yinit == 1:          # Threshold
            s = np.zeros((m,1))
            iits = 0
            while math.pow(np.linalg.norm(s),2) < (float(rho)/n):                
                y = np.zeros((n,1))                     
                y[idx] = 1
                s = np.dot(A,y)
                if k>0:       # python is zero-based             
                    s = s - (np.dot(xsav,(np.multiply(dsav,(np.dot(ysav.T,y))))))                    
                    
                idx = np.mod(idx, n) + 1
                if idx == n:        # When idx reaches n it should be changed to zero (otherwise an index out of bounds error will occur)
                    idx = 0
                iits = iits + 1
            iitssav[k] = iits
        elif yinit == 2:        # Cycling Periodic Ones
            y = np.zeros((n,1))
            index = np.mod(k-1,n)+1
            if index < n:                 
                y[index] = 1
            else:
                y[0] = 1   
        elif yinit == 3:        # All Ones
            y = np.ones((n,1))
        elif yinit == 4:        # Periodic Ones
            y = np.zeros((n,1))
            ii = np.arange(0,n,100)
            for i in ii: # python is zero-based
                y[i] = 1 
        else:
            try:
                pass
            except ValueError:
                print 'Invalid choice for C.'
                
        # Inner loop
        
        for l in range (0,lmax):
	    print "l: ",l
            # Fix y and Solve for x
            
            s = np.dot(A,y)            
            if k > 0:       # python is zero-based
                s = s - (np.dot(xsav,(np.multiply(dsav,(np.dot(ysav.T,y))))))

            [x, xcnt, _] = sddsolve(s, m)

            # Fix x and Solve for y
            
            s = np.dot(A.T,x)
            if k > 0:
                s = s - (np.dot(ysav,(np.multiply(dsav,(np.dot(xsav.T,x))))))
            
            [y, ycnt, fmax] = sddsolve(s, n)
            
            # Check Progress
            
            d = np.sqrt(fmax * ycnt) / (ycnt * xcnt)

            beta = math.pow(d,2) * ycnt * xcnt
            
            if l > 0: # python is zero-based
                alpha = (beta - betabar) / betabar
                if alpha <= alphamin:
                    break
            
            betabar = beta
        
        # Save
        
        xsav[:, k] = x.T            # shape conflict (matlab deals with this internally)        
        ysav[:, k] = y.T
        dsav[k, 0] = d              # python is zero-based        
        rho = max([rho-beta,0])
        rhosav[k] = rho
        itssav[k] = l
        
        # Threshold Test
        
        if rho <= rhomin:
            break
        
    return dsav, xsav, ysav, itssav, rhosav, iitssav

def sddsolve(s, m):
    x = np.zeros((m,1))   
    x = np.asarray(x)
    
    for i in range(0,m):        # python is zero-based
        if s[i] < 0:
            x[i,0] = -1         # python is zero-based
            s[i] = -s[i]
        else:
            x[i,0] = 1          # python is zero-based
    
    sorted_array =sorted(enumerate(-s), key=operator.itemgetter(1)) # Sort array and get index of original unsorted data
    sorted_array = np.asarray(sorted_array)
    sorts = -sorted_array[:,1]
    indexsort = sorted_array[:,0]
    
    f = np.zeros((m))
    f = np.asfarray(f)
    f[0] = sorts[0]             # python is zero-based
    for i in range(1,m):
        f[i] = sorts[i] + f[i-1]
    
    f = np.divide(np.power(f,2),np.arange(1,m+1,1))
    
    imax = 0                    # 1 will be added later on
    fmax = f[0]                 # python is zero-based
    for i in range(1,m):
        if f[i] >= fmax:
            imax = i        
            fmax = f[i]
    
    for i in range(imax+1,m):
        x[int(indexsort[i])] = 0
        
    imax += 1                   # + 1 to correct imax

    return x, imax, fmax 
def full_connect_layer(n):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = 'InnerProduct'
    layer.inner_product_param.num_output = n
    return layer

def conv_layer(h, w, n, group=1, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = 'Convolution'
    if (h == w):
        layer.convolution_param.kernel_size.append(h)
    else:
        layer.convolution_param.kernel_h = h
        layer.convolution_param.kernel_w = w
    layer.convolution_param.num_output = n
    if (group != 1):
        layer.convolution_param.group = group
    if (pad_h != 0 or pad_w != 0):
        layer.convolution_param.pad_h = pad_h
        layer.convolution_param.pad_w = pad_w
    if (stride_h != 1 or stride_w != 1):
        layer.convolution_param.stride_h = stride_h
        layer.convolution_param.stride_w = stride_w
    return layer


def find_layer_by_name(model, layer_name):
    k = 0
    while model.layer[k].name != layer_name:
        k += 1
        if (k > len(model.layer)):
            raise IOError('layer with name %s not found' % layer_name)
    return k

def find_layer_group(model, layer_name):
    k=0
    group = 1
    while model.layer[k].name != layer_name:
	k +=1
	group = model.layer[k].convolution_param.group
	if (k > len(model.layer)):
            raise IOError('layer with name %s not found' % layer_name)
    return group
 
def create_deploy_model(model, input_dim=[64, 3, 32, 32]):
    new_model = caffe.proto.caffe_pb2.NetParameter()
    new_model.input.extend(['data'])
    new_model.input_dim.extend(input_dim)
    for i in range(2,len(model.layer)-2):
	if model.layer[i].name.find("loss")==-1:
           new_model.layer.extend([model.layer[i]])
    return new_model
def create_deploy_model_sdd(model, input_dim=[64, 3, 32, 32]):
    new_model = caffe.proto.caffe_pb2.NetParameter()
    new_model.input.extend(['data'])
    new_model.input_dim.extend(input_dim)
    new_model.input.extend(['im_info'])
    new_model.input_dim.extend([1,3,0,0])
    for i in range(0,len(model.layer)):
        new_model.layer.extend([model.layer[i]])
    return new_model
    
def load_model(filename):
    model = caffe.proto.caffe_pb2.NetParameter()
    input_file = open(filename, 'r')
    google.protobuf.text_format.Merge(str(input_file.read()), model)
    input_file.close()
    return model

def save_model(model, filename):
    output_file = open(filename, 'w')
    google.protobuf.text_format.PrintMessage(model, output_file)
    output_file.close()

def find_layer_type(model,layer_name):
    k = 0
    print layer_name
    layer_type = 'Convolution'
    while model.layer[k].name != layer_name:
        k += 1
	layer_type = model.layer[k].type
	if (k > len(model.layer)):
		raise IOError('layer with name % not found ' % layer_name)
    return layer_type
def accelerate_model(model, K, layer_to_decompose):
    k = layer_to_decompose
    new_model = caffe.proto.caffe_pb2.NetParameter()
    for i in range(k):
	    new_model.layer.extend([model.layer[i]]) 
    decomposed_layer = model.layer[k]
    if decomposed_layer.type not in  ['Convolution','InnerProduct']:
       raise AttributeError('only convolution and innerproduct layer can be decomposed')

    if decomposed_layer.type == 'Convolution':
      param = decomposed_layer.convolution_param   
      print "h: ", param.kernel_h
      print "size: ", param.kernel_size[0]
      print "group: ",param.group
      print "num_output: ", param.num_output
      if not hasattr(param, 'pad'):
          param.pad = [0]
      if param.pad == []:
          param.pad.append(0)
      if not hasattr(param, 'stride'):
          param.stride = [1]
      #conv_layer(h,w,n,group,pad_h,pad_w,stride_h,stride_w)
      new_model.layer.extend([conv_layer(param.kernel_size[0],param.kernel_size[0], K ,param.group, pad_h=param.pad[0],pad_w=param.pad[0])])#, stride_h=param.stride[0],stride_w=param.stride[0])])
      #new_model.layer.extend([conv_layer(param.kernel_size[0],param.kernel_size[0],param.num_output,param.group, pad_h=param.pad[0], stride_h=param.stride[0])])
      new_model.layer.extend([conv_layer(1,1,param.num_output)])
    elif decomposed_layer.type == 'InnerProduct' :
      param = decomposed_layer.inner_product_param
      new_model.layer.extend([full_connect_layer(K)])
      new_model.layer.extend([full_connect_layer(param.num_output)])

    name = decomposed_layer.name
    for i in range(2):
        new_model.layer[k+i].name = name + '-' + str(i + 1)
        new_model.layer[k+i].bottom.extend([name + '-' + str(i)])
        new_model.layer[k+i].top.extend([name + '-' + str(i + 1)])
    new_model.layer[k].bottom[0] = model.layer[k].bottom[0]
    new_model.layer[k+1].top[0] = model.layer[k].top[0]
    for i in range(k+1, len(model.layer)):
        new_model.layer.extend([model.layer[i]])
    return new_model
     

def prepare_models(LAYER, K, NET_PATH, NET_NAME, INPUT_DIM):
    PATH = NET_PATH
    NET_PREFIX = PATH + NET_NAME
    input_dim = INPUT_DIM
    model = load_model(NET_PREFIX + '.prototxt')
    ind = find_layer_by_name(model, LAYER)
    new_model = accelerate_model(model, K, ind)
    save_model(new_model, NET_PREFIX + '_accelerated.prototxt')
    new_deploy = create_deploy_model(new_model, input_dim)
    save_model(new_deploy, NET_PREFIX + '_accelerated_deploy.prototxt')
    deploy = create_deploy_model(model, input_dim)
    save_model(deploy, NET_PREFIX + '_deploy.prototxt')

    net = caffe.Classifier(NET_PREFIX + '_deploy.prototxt', NET_PREFIX + '.caffemodel')
    fast_net = caffe.Classifier(NET_PREFIX + '_accelerated_deploy.prototxt', NET_PREFIX + '.caffemodel')
    l = ind - 1#layer index in deploy version
#    l = ind + 1 
    print("l:",l)
    weights = net.layers[l].blobs[0].data
    bias = net.layers[l].blobs[1]

    print "weights shape",weights.shape
    if find_layer_type(model, LAYER) == 'Convolution':
      if find_layer_group(model, LAYER) >= 1:
        T = dtensor(weights)
        n = weights.shape[0]
        c = weights.shape[1]
        h = weights.shape[2]
        w = weights.shape[3]
        A = np.zeros((c*h*w,n))
        for i in range(n):
            for j in range(c):
               for k in range(h):
                   for m in range(w):
                       A[j*h*w+k*h+m,i]=T[i,j,k,m]
        D, X, Y, _, _, _, = sdd(A=A,kmax=K)
        for i in range(K):
            for j in range(c*h*w):
                X[j,i]=X[j,i]*D[i,0]
        print "X shape: ", X.shape
        print "Y shape: ", Y.shape
        print "A shape: ", A.shape
        #print np.core.multiarray.dot(X,Y.T)
        #print np.core.multiarray.dot(X,Y.T)-A
        X1 = np.zeros((K,c,h,w))
        Y1 = np.zeros((n,K,1,1))
        for i in range(K):
           for j in range(c):
              for k in range(h):
                 for m in range(w):
                    X1[i,j,k,m]=X[m+k*w+j*h*w,i]
                    #print "m+k*m+j*k*m: ", m+k*m+j*k*m
                    #print "i", i
        for i in range(n):
           for j in range(K):
               Y1[i,j,0,0]=Y[i,j]
  
        zeros = 0
        for i in range(K):
           for j in range(c*h*w):
               if X[j, i] == 0:
                 zeros = zeros +1 
        sparse = float(zeros)/(K*c*h*w)
        print "num_of_zeroX/total_num=",zeros,"/",K*c*h*w,"=", sparse,'\n'
        zeros1 = 0
        for i in range(n):
           for j in range(K):
              if Y[i,j] == 0:
                zeros1 = zeros1 + 1
        sparse = float(zeros1)/(K*n)
        print "num_of_zeroY/total_num1=",zeros1,"/",K*n,"=",sparse,'\n'
        print "sparse_ratio: ",float(zeros+zeros1)/(K*c*h*w+K*n)
        print "X1: ",X1.shape 
	print "Y1: ",Y1.shape
        np.copyto(fast_net.layers[l].blobs[0].data, X1)
        np.copyto(fast_net.layers[l+1].blobs[0].data, Y1)
        np.copyto(fast_net.layers[l+1].blobs[1].data, bias.data)
      elif find_layer_group(model, LAYER) <1:
        T =  dtensor(weights)
	n = weights.shape[0]
        c = weights.shape[1]
        h = weights.shape[2]
        w = weights.shape[3]
	group = find_layer_group(model, LAYER)
	c1 = c/group
	n1 = n/group
	A1 = np.zeros((c*h*w/group, n/group))
	A2 = np.zeros((c*h*w/group, n/group))
        for i in range(n1):
            for j in range(c1):
               for k in range(h):
                   for m in range(w):
                       A1[j*h*w+k*h+m,i]=T[i,j,k,m]
        for i in range(n1,n):
            for j in range(c1,c):
               for k in range(h):
                   for m in range(w):
                       A1[j*h*w+k*h+m,i]=T[i,j,k,m]
        D1, X1, Y1, _, _, _, = sdd(A=A1,kmax=K)
        D2, X2, Y2, _, _, _, = sdd(A=A2,kmax=K)	
        for i in range(K):
            for j in range(c1*h*w):
                X1[j,i]=X1[j,i]*D1[i,0]
        for i in range(K):
            for j in range(c1*h*w):
                X2[j,i]=X2[j,i]*D2[i,0]

        print "X1 shape: ", X1.shape
        print "Y1 shape: ", Y1.shape
        print "A1 shape: ", A1.shape
        print "X2 shape: ", X2.shape
        print "Y2 shape: ", Y2.shape
        print "A2 shape: ", A2.shape
        X = np.zeros((K,c,h,w))
        Y = np.zeros((n,K,1,1))
        for i in range(K):
           for j in range(c1):
              for k in range(h):
                 for m in range(w):
                    X1[i,j,k,m]=X1[m+k*w+j*h*w,i]
        for i in range(K):
          for j in range(c1):
             for k in range(h):
                for m in range(w):
                   X1[i,j,k,m]=X1[m+k*w+(j+c1)*h*w,i]
        for i in range(n1):
          for j in range(K):
              Y[i,j,0,0]=Y1[i,j]
        for i in range(n1):
         for j in range(K):
             Y[i+n1,j,0,0]=Y2[i,j]
      
        zeros = 0
        for i in range(K):
           for j in range(c1*h*w):
               if X1[j, i] == 0:
                 zeros = zeros +1 
        for i in range(K):
           for j in range(c1*h*w):
               if X2[j, i] == 0:
                 zeros = zeros +1 

        sparse = float(zeros)/(K*c*h*w)
        print "num_of_zeroX/total_num=",zeros,"/",K*c*h*w,"=", sparse,'\n'
        zeros1 = 0
        for i in range(n1):
           for j in range(K):
              if Y1[i,j] == 0:
                zeros1 = zeros1 + 1
        for i in range(n1):
           for j in range(K):
              if Y2[i,j] == 0:
                zeros1 = zeros1 + 1
        sparse = float(zeros1)/(K*n)
        print "num_of_zeroY/total_num1=",zeros1,"/",K*n,"=",sparse,'\n'
        print "sparse_ratio: ",float(zeros+zeros1)/(K*c*h*w+K*n)
        np.copyto(fast_net.layers[l].blobs[0].data, X)
        np.copyto(fast_net.layers[l+1].blobs[0].data, Y)
        np.copyto(fast_net.layers[l+1].blobs[1].data, bias.data)
 
    elif find_layer_type(model, LAYER) == 'InnerProduct':
      T = dtensor(weights)
      print T.shape
      D, X, Y, _, _, _, = sdd(A=T,kmax=K)
      print "X shape", X.shape
      print "Y shape", Y.shape
      for i in range(K):
	 for j in range(T.shape[0]):
           X[j,i]=X[j,i]*D[i,0]
      zeros = 0
      for i in range(X.shape[0]):
         for j in range(X.shape[1]):
             if X[i, j] == 0:
               zeros = zeros +1 
      sparse = float(zeros)/(X.shape[1]*X.shape[0])
      print "num_of_zeroX/total_num=",zeros,"/",X.shape[0]*X.shape[1],"=", sparse,'\n'
      zeros1 = 0
      for i in range(Y.shape[0]):
         for j in range(Y.shape[1]):
            if Y[i,j] == 0:
              zeros1 = zeros1 + 1
      sparse = float(zeros1)/(Y.shape[1]*Y.shape[0])
      print "num_of_zeroY/total_num1=",zeros1,"/",Y.shape[0]*Y.shape[1],"=",sparse,'\n'
      print "sparse_ratio: ",float(zeros+zeros1)/(Y.shape[0]*Y.shape[1]+X.shape[0]*X.shape[1])

      np.copyto(fast_net.layers[l].blobs[0].data, Y.T)
      np.copyto(fast_net.layers[l+1].blobs[0].data, X)
      np.copyto(fast_net.layers[l+1].blobs[1].data, bias.data)

    else:
	    print("!!!!!!!!!! not type allowed")


   
    fast_net.save(NET_PREFIX + '_accelerated.caffemodel')
