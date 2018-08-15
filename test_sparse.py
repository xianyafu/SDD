import numpy as np, math, operator 
def sparse_weights(T,n,c,h,w):
  a = T.reshape(-1)
  abs_count = np.ones(shape=a.shape)

  s = 0
  for i in range(0,a.size):
    if a[i]==0:
	 s = s+1
  print(s,"/",a.size," ",s/a.size)

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
  print(s2,"/",a.size," ",float(s2/a.size))
  print('------------oral----------')
  print(T)
  print('------------sparse--------')
  print(new_data)
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
  print(s,"/",a.size," ",s/a.size)
  print(s1,"/",a.size," ",float(s1/a.size))

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
  print(s2,"/",a.size," ",float(s2/a.size))
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

def test_sparse():
    a = np.ones(shape=[2,2,2,2])
    q = 0
    b = np.ones(shape=[2,8])
    for i in range(a.shape[0]):
       for j in range(a.shape[1]):
	    for k in range(a.shape[2]):
		 for l in range(a.shape[3]):
		     a[i][j][k][l]=q
		     q = q+1
    for i in range(b.shape[0]):
       for j in range(b.shape[1]):
            b[i][j]=q
	    q=q+1
    print(a)
    W = sparse_weights(a,2,2,2,2)
    print(a)
    print(W)
    print(b)
    X = sparse_weights_2(b,2,8)
    print(X)
test_sparse()
