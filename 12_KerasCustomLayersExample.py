
# coding: utf-8

# # Using Keras Custom Layers

# Example code for the lecture series "Machine Learning for Physicists" by Florian Marquardt
# 
# Lecture 12
# 
# See https://machine-learning-for-physicists.org and the current course website linked there!
# 
# This notebook is distributed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license:
# 
# https://creativecommons.org/licenses/by-sa/4.0/

# This notebook shows how to:
# - construct a keras custom layer (for a simple convolutional layer that respects periodic boundary conditions in 1D)
# 

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential # Sequential is the neural-network class
from tensorflow.keras.layers import Lambda # Dense is the standard network layer

# array math:
import numpy as np

# plotting:
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# for updating display 
# (very simple animation)
from IPython.display import clear_output
from time import sleep


# In[2]:


class PeriodicConvolution(keras.layers.Layer):
    # init gets called when setting up the network
    def __init__(self,kernel_size=3,**kwargs):
        self.kernel_size=kernel_size
        super(PeriodicConvolution, self).__init__(**kwargs)

    # build gets called when the network is first evaluated
    # that means, the size of the input is now known
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[self.kernel_size],
            initializer="random_normal",
            trainable=True,
        )
        
    # call gets called whenever the network is evaluated
    # (actually it may get called only once, to build up the
    # symbolic graph, but that is a detail)
    def call(self, inputs):
        j0=int((self.kernel_size-1)/2)
        # do convolution using tf.roll, which respects periodic
        # boundary conditions!
        # Note: unlike numpy, we cannot simply assign a zeros array first,
        # because we are not allowed to assign afterwards to the Tensor,
        # so we just initialize by treating j==0 separately
        for j in range(self.kernel_size):
            if j==0:
                z=self.w[j]*tf.roll(inputs,shift=j-j0,axis=1)
            else:
                z+=self.w[j]*tf.roll(inputs,shift=j-j0,axis=1)
        return z


# In[3]:


Net=Sequential()
Net.add(PeriodicConvolution(kernel_size=3))

Net.compile(loss='mean_square_error', optimizer='adam')


# In[4]:


y_in=np.array([[0.,0.,3.,0.,0.]])


# In[5]:


y_out=Net.predict_on_batch(y_in)
print(y_out)


# In[6]:


Net.layers[0].w


# In[7]:


Net.layers[0].w.assign(np.array([-1,0,1]))


# In[8]:


y_out=Net.predict_on_batch(y_in)
print(y_out)


# In[9]:


y_out=Net.predict_on_batch(np.array([[0.,0.,0.,0.,3.]]))
print(y_out)

