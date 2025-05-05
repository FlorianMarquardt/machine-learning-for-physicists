
# coding: utf-8

# # Homework: Boltzmann Machines applied to MNIST
# 
# This shows how to train a Boltzmann machine, to sample from an observed probability distribution. It uses the MNIST digits images that are included in every keras installation.

# Example code for the lecture series "Machine Learning for Physicists" by Florian Marquardt
# 
# Lecture 9, Homework (this is discussed in session 10)
# 
# See https://machine-learning-for-physicists.org and the current course website linked there!
# 
# This notebook is distributed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license:
# 
# https://creativecommons.org/licenses/by-sa/4.0/

# This notebook shows how to:
# - use a Boltzmann machine to sample from an observed high-dimensional probability distribution (e.g. produce images that look similar to observed training images); applied to the case of MNIST
# 
# It also implements some of the tricks discussed by the inventor of RBM, G. Hinton, in
# 
# https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

import tensorflow as tf

from IPython.display import clear_output
from time import sleep


# In[2]:


def BoltzmannStep(v,b,w,do_random_sampling=True):
    """
    Perform a single step of the Markov chain,
    going from visible units v to hidden units h,
    according to biases b and weights w.
    
    z_j = b_j + sum_i v_i w_ij
    
    and P(h_j=1|v) = 1/(exp(-z_j)+1)
    
    Note: you can go from h to v, by inserting
    instead of v the h, instead of b the a, and
    instead of w the transpose of w
    """
    batchsize=np.shape(v)[0]
    hidden_dim=np.shape(w)[1]
    z=b+np.dot(v,w)
    P=1/(np.exp(-z)+1)
    # now, the usual trick to obtain 0 or 1 according
    # to a given probability distribution:
    # just produce uniform (in [0,1]) random numbers and
    # check whether they are below the cutoff given by P
    if do_random_sampling:
        p=np.random.uniform(size=[batchsize,hidden_dim])
        return(np.array(p<=P,dtype='int'))
    else:
        return(P) # no binary random output, just the prob. distribution itself!
    
def BoltzmannSequence(v,a,b,w,drop_h_prime=False,do_random_sampling=True,
                      do_not_sample_h_prime=False,
                     do_not_sample_v_prime=False):
    """
    Perform one sequence of steps v -> h -> v' -> h'
    of a Boltzmann machine, with the given
    weights w and biases a and b!
    
    All the arrays have a shape [batchsize,num_neurons]
    (where num_neurons is num_visible for v and
    num_hidden for h)
    
    You can set drop_h_prime to True if you want to
    use this routine to generate arbitrarily long sequences
    by calling it repeatedly (then don't use h')
    Returns: v,h,v',h'
    """
    h=BoltzmannStep(v,b,w,do_random_sampling=do_random_sampling)
    if do_not_sample_v_prime:
        v_prime=BoltzmannStep(h,a,np.transpose(w),do_random_sampling=False)
    else:
        v_prime=BoltzmannStep(h,a,np.transpose(w),do_random_sampling=do_random_sampling)
    if not drop_h_prime:
        if do_not_sample_h_prime: # G. Hinton recommends not sampling in the v'->h' step (reduces noise)
            h_prime=BoltzmannStep(v_prime,b,w,do_random_sampling=False)
        else:
            h_prime=BoltzmannStep(v_prime,b,w,do_random_sampling=do_random_sampling)
    else:
        h_prime=np.zeros(np.shape(h))
    return(v,h,v_prime,h_prime)

def trainStep(v,a,b,w,do_random_sampling=True,do_not_sample_h_prime=False,
             do_not_sample_v_prime=False):
    """
    Given a set of randomly selected training samples
    v (of shape [batchsize,num_neurons_visible]), 
    and given biases a,b and weights w: update
    those biases and weights according to the
    contrastive-divergence update rules:
    
    delta w_ij = eta ( <v_i h_j> - <v'_i h'_j> )
    delta a_i  = eta ( <v_i> - <v'_i>)
    delta b_j  = eta ( <h_j> - <h'_j>)
    
    Returns delta_a, delta_b, delta_w, but without the eta factor!
    It is up to you to update a,b,w!
    """
    v,h,v_prime,h_prime=BoltzmannSequence(v,a,b,w,do_random_sampling=do_random_sampling,
                                         do_not_sample_h_prime=do_not_sample_h_prime,
                                         do_not_sample_v_prime=do_not_sample_v_prime)
    return( np.average(v,axis=0)-np.average(v_prime,axis=0) ,
            np.average(h,axis=0)-np.average(h_prime,axis=0) ,
            np.average(v[:,:,None]*h[:,None,:],axis=0)-
               np.average(v_prime[:,:,None]*h_prime[:,None,:],axis=0) )


# In[3]:


def produce_sample_images(batchsize,num_visible,x_train,threshold=0.7,do_digitize=True):
    """
    Produce 'batchsize' samples, of length num_visible.
    Returns array v of shape [batchsize,num_visible]
    """
    j=np.random.randint(low=0,high=np.shape(x_train)[0],size=batchsize) # pick random samples
    
    # reshape suitably, and digitize (so output is 0/1 values)
    if do_digitize:
        return( np.array( np.reshape(x_train[j,:,:],[batchsize,num_visible])>threshold, dtype='int' ) )
    else:
        return(  np.reshape(x_train[j,:,:],[batchsize,num_visible]) )


# In[4]:


# Load the MNIST data using tensorflow/keras
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data(path="mnist.npz")

x_train=x_train/256.


# In[5]:


# Show the shape of these arrays
print(np.shape(x_train),np.shape(y_train),np.shape(x_test),np.shape(y_test))


# In[6]:


# display a few images, for fun
nimages=10
fig,ax=plt.subplots(ncols=nimages,nrows=1,figsize=(nimages,1))
for n in range(nimages):
    ax[n].imshow(x_train[n,:,:])
    ax[n].set_title(str(y_train[n])) # the label
    ax[n].axis('off')
plt.show()


# In[7]:


# pick a few random samples:
nimages=7
Npixels=28
samples=produce_sample_images(batchsize=nimages,num_visible=Npixels**2,x_train=x_train)
# now unpack them again and display them:
fig,ax=plt.subplots(ncols=nimages,nrows=1,figsize=(nimages,1))
for n in range(nimages):
    ax[n].imshow(np.reshape(samples[n,:],[Npixels,Npixels]))
    ax[n].axis('off')
plt.show()


# In[8]:


# a little trick (useful later): show them all at once, in one imshow
# some weird reshape/transpose gymnastics, found by trial and error
plt.imshow(np.transpose(np.reshape(np.transpose(np.reshape(samples,[nimages,Npixels,Npixels]),
                                                axes=[0,2,1]),[nimages*Npixels,Npixels])))
plt.axis('off')
plt.show()


# ## Basic training, according to the simple principle of an RBM presented in the lecture (all units are binary 0 or 1 all the time, randomly chosen according to the calculated probability distribution)

# In[92]:


# Now: the training
# here: purely using random binary sampling of all
# units at all times (this is not the most efficient way,
# but implements directly the basic principle shown in the lecture)

Npixels=28
num_visible=Npixels**2
num_hidden=60
batchsize=50
eta=0.1
nsteps=10000
skipsteps=10

a=np.random.randn(num_visible)
b=np.random.randn(num_hidden)
w=0.01*np.random.randn(num_visible,num_hidden)

# test_samples=np.zeros([num_visible,nsteps])

for j in range(nsteps):
    v=produce_sample_images(batchsize,num_visible,x_train)
    da,db,dw=trainStep(v,a,b,w)
    a+=eta*da
    b+=eta*db
    w+=eta*dw
    print("{:05d} / {:05d}".format(j,nsteps),end="   \r")


# In[93]:


# Now: visualize the typical samples generated (from some starting point)
# run several times to continue this. It basically is a random walk
# through the space of all possible configurations, hopefully according
# to the probability distribution that has been trained!

nsteps=1000
num_samples=20
test_samples=np.zeros([num_samples,num_visible])
skipsteps=1
substeps=400 # how many steps to take before showing a new picture

v_prime=np.zeros(num_visible)
h=np.zeros(num_hidden)
h_prime=np.zeros(num_hidden)

for j in range(nsteps):
    for k in range(substeps):
        v,h,v_prime,h_prime=BoltzmannSequence(v,a,b,w,drop_h_prime=True) # step from v via h to v_prime!
    test_samples[j%num_samples,:]=v[0,:]
    v=np.copy(v_prime) # use the new v as a starting point for next step!
    if j%skipsteps==0 or j==nsteps-1:
        clear_output(wait=True)
        plt.imshow(np.transpose(np.reshape(np.transpose(np.reshape(test_samples,[num_samples,Npixels,Npixels]),
                                                axes=[0,2,1]),[num_samples*Npixels,Npixels])),
                  interpolation='none')
        plt.axis('off')
        plt.show()


# ## More advanced training: do not randomly sample h' and v'

# In[29]:


# Now: the training
#
# Here we use the more sophisticated approach, where
# h' and v' are not binary (not randomly sampled), rather
# they are taken as the probability distribution (numbers
# between 0 and 1). This is a trick recommend by G. Hinton
# in his review of Boltzmann Machines. It effectively means
# less sampling noise.
#
# Also, we initialize the biases and weights according to the
# tricks given in that review!

Npixels=28
num_visible=Npixels**2
num_hidden=60
batchsize=10
eta=0.0001
nsteps=10*30000
skipsteps=10

# get average brightness of training images:
p_avg=np.average(np.reshape(x_train,[np.shape(x_train)[0],Npixels**2]),axis=0)
a=np.log(p_avg/(1.0+1e-6-p_avg)+1e-6) # recipe for visible biases
b=np.zeros(num_hidden) # recipe for hidden biases
w=0.01*np.random.randn(num_visible,num_hidden) # recipe for weights

# test_samples=np.zeros([num_visible,nsteps])

for j in range(nsteps):
    v=produce_sample_images(batchsize,num_visible,x_train,
                           do_digitize=False)
    da,db,dw=trainStep(v,a,b,w,
                      do_not_sample_h_prime=True,
                       do_not_sample_v_prime=True)
    a+=eta*da
    b+=eta*db
    w+=eta*dw
    print("{:06d} / {:06d}".format(j,nsteps),end="   \r")



# In[32]:


# Now: visualize the typical samples generated (from some starting point)
# run several times to continue this. It basically is a random walk
# through the space of all possible configurations, hopefully according
# to the probability distribution that has been trained!

nsteps=20
num_samples=20
test_samples=np.zeros([num_samples,batchsize,num_visible])
test_hidden=np.zeros([num_samples,batchsize,num_hidden])
skipsteps=1
substeps=400 # how many steps to take before showing a new picture

v_prime=np.zeros([batchsize,num_visible])
h=np.zeros([batchsize,num_hidden])
h_prime=np.zeros([batchsize,num_hidden])

v=produce_sample_images(batchsize,num_visible,x_train,
                       do_digitize=False)
    
for j in range(nsteps):
    for k in range(substeps):
        v,h,v_prime,h_prime=BoltzmannSequence(v,a,b,w,
                                  drop_h_prime=True,
                                  do_not_sample_v_prime=True) # step from v via h to v_prime!
    test_samples[j%num_samples,:,:]=v[:,:]
    test_hidden[j%num_samples,:]=h[:,:]
    v=np.copy(v_prime) # use the new v as a starting point for next step!
    if j%skipsteps==0 or j==nsteps-1:
        clear_output(wait=True)
        fig,ax=plt.subplots(ncols=1,nrows=batchsize,figsize=(num_samples,batchsize))
        for n in range(batchsize):
            ax[n].imshow(np.transpose(np.reshape(np.transpose(np.reshape(test_samples[:,n,:],[num_samples,Npixels,Npixels]),
                                                    axes=[0,2,1]),[num_samples*Npixels,Npixels])),
                      interpolation='none')
            ax[n].axis('off')
        plt.show()
        fig,ax=plt.subplots(ncols=1,nrows=batchsize,figsize=(num_samples,batchsize))
        for n in range(batchsize):
            ax[n].imshow(np.transpose(np.reshape(np.transpose(np.reshape(test_hidden[:,n,:],[num_samples,6,10]),
                                                    axes=[0,2,1]),[num_samples*10,6])),
                      interpolation='none')
            ax[n].axis('off')
        plt.show()        

