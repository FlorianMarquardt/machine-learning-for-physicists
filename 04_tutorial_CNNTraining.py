
# coding: utf-8

# # Visualizing the training of Convolutional Neural Networks (with keras)

# Example code for the lecture series "Machine Learning for Physicists" by Florian Marquardt
# 
# Lecture 4, Tutorials
# 
# See https://machine-learning-for-physicists.org and the current course website linked there!

# This notebook shows how to:
# - visualize the training of convolutional autoencoders using keras
# 
# The networks are 2D convolutional networks, with the same input and output dimensions, and a bottleneck layer in the middle.
# 
# You define the network and the type of images that are generated for training – this notebook will help you visualize the training evolution.

# ### Imports: numpy and matplotlib and keras

# In[117]:


# keras: Sequential is the neural-network class, Dense is
# the standard network layer
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras import optimizers # to choose more advanced optimizers like 'adam'

import numpy as np

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# for subplots within subplots:
from matplotlib import gridspec

# for nice inset colorbars: (approach changed from lecture 1 'Visualization' notebook)
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# for updating display 
# (very simple animation)
from IPython.display import clear_output
from time import sleep


# ### Functions

# In[189]:


# visualization routines:

def visualize_CNN_training(network,
                               image_generator, resolution,
                    steps=100, batchsize=10,
                              visualize_nsteps=1, plot_img_pixels=3,
                          plot_img_cols=10,
                          plot_img_rows=5,
                          show_intermediate_layers=True):
    """
    Visualize the training of a (2D) convolutional neural network autoencoder.
    
    network is the network that you have defined using keras.
    
    'resolution' (called M below) is the image resolution in pixels
    
    image_generator is the name of a function that
    is called like
        image_generator(batchsize,x,y)
    and which has to return an array of shape
        [batchsize,M,M]
    that contains randomly generated MxM images (e.g. randomly
    placed circles or whatever you want to consider). The
    MxM arrays x and y are already filled with coordinates between -1 and 1.
    
    An example that returns images of randomly placed circles:
    
    def my_generator(batchsize,x,y):
        R=np.random.uniform(size=batchsize)
        x0=np.random.uniform(size=batchsize,low=-1,high=1)
        y0=np.random.uniform(size=batchsize,low=-1,high=1)
        return( 1.0*((x[None,:,:]-x0[:,None,None])**2 + (y[None,:,:]-y0[:,None,None])**2 < R[:,None,None]**2) )

   
    steps is the number of training steps
    
    batchsize is the number of samples per training step
            
    visualize_n_steps>1 means skip some steps before
    visualizing again (can speed up things)
    
    show_intermediate_layers==True means show the intermediate activations.
    Otherwise show the weights!
    
    These are always shown in the upper left corner, as a tiled image,
    whose properties are determined by:
        plot_img_pixels: the resolution for each of the image tiles
        plot_img_cols  : the number of columns of images
        plot_img_rows  : the number of rows of images
    Images (activations or weights) that are larger will be cut off.
    If there are more images than fit, the rest will be left out.
    The lowest layer starts in the bottom left. For activations, for
    each layer one runs through all the channels, and then the images
    for the next layer will start. Likewise for weights.
    """
    global y_target # allow access to target from outside
    
    M=resolution
        
    vals=np.linspace(-1,1,M)
    x,y=np.meshgrid(vals,vals)
    
    y_test=np.zeros([1,M,M,1])
    y_test[:,:,:,0]=image_generator(1,x,y)
    
    y_in=np.zeros([batchsize,M,M,1])

    costs=np.zeros(steps)
    extractor=get_layer_activation_extractor(network)
    
    for j in range(steps):
        # produce samples:
        y_in[:,:,:,0]=image_generator(batchsize,x,y)
        y_target=np.copy(y_in) # autoencoder wants to reproduce its input!
        
        # do one training step on this batch of samples:
        costs[j]=network.train_on_batch(y_in,y_target)
        
        # now visualize the updated network:
        if j%visualize_nsteps==0:
            clear_output(wait=True) # for animation
            if j>10:
                cost_max=np.average(costs[0:j])*1.5
            else:
                cost_max=costs[0]
            
            # nice layout (needs matplotlib v3)
            fig=plt.figure(constrained_layout=True,figsize=(8,4))
            gs=fig.add_gridspec(ncols=8,nrows=4)
            filter_plot=fig.add_subplot(gs[0:3,0:4])
            cost_plot=fig.add_subplot(gs[3,0:4])
            test_in_plot=fig.add_subplot(gs[0:2,4:6])
            test_out_plot=fig.add_subplot(gs[0:2,6:8])

            cost_plot.plot(costs)
            cost_plot.set_ylim([0,cost_max])
            
            # test the network on a fixed test image!
            y_test_out=network.predict_on_batch(y_test)
            test_in_plot.imshow(y_test[0,:,:,0],origin='lower')
            test_out_plot.imshow(y_test_out[0,:,:,0],origin='lower')
            test_in_plot.axis('off')
            test_out_plot.axis('off')
            
            if show_intermediate_layers:
                features=extractor(y_test)
                n1=0; n2=0
                max_n1=plot_img_rows
                max_n2=plot_img_cols
                pix=plot_img_pixels
                img=np.full([(pix+1)*max_n1,(pix+1)*max_n2],1.0)
                for feature in features:
                    for m in range(feature.shape[-1]):
                        w=feature[0,:,:,m]
                        ws=np.shape(w)
                        if n1<max_n1 and n2<max_n2:
                            W=np.zeros([pix,pix])
                            if ws[0]<pix:
                                W[0:ws[0],0:ws[0]]=w[:,:]
                            else:
                                W[:,:]=w[0:pix,0:pix]                            
                            img[n1*(pix+1):(n1+1)*(pix+1)-1,n2*(pix+1):(n2+1)*(pix+1)-1]=W
                            n2+=1
                            if n2>=max_n2:
                                n2=0
                                n1+=1                
            else: # rather, we want the weights! (filters)
                n1=0; n2=0
                max_n1=plot_img_rows
                max_n2=plot_img_cols
                pix=plot_img_pixels
                img=np.zeros([(pix+1)*max_n1,(pix+1)*max_n2])
                for ly in network.layers:
                    w=ly.get_weights()
                    if w!=[]:
                        w=w[0]
                        ws=np.shape(w)
                        for k1 in range(ws[2]):
                            for k2 in range(ws[3]):
                                if n1<max_n1 and n2<max_n2:
                                    W=np.zeros([pix,pix])
                                    if ws[0]<pix:
                                        W[0:ws[0],0:ws[0]]=w[:,:,k1,k2]
                                    else:
                                        W[:,:]=w[0:pix,0:pix,k1,k2]                            
                                    img[n1*(pix+1):(n1+1)*(pix+1)-1,n2*(pix+1):(n2+1)*(pix+1)-1]=W
                                    n2+=1
                                    if n2>=max_n2:
                                        n2=0
                                        n1+=1
            filter_plot.imshow(img,origin='lower')
            filter_plot.axis('off')
            plt.show()
    print("Final cost value (averaged over last 50 batches): ", np.average(costs[-50:]))


def print_layers(network, y_in):
    """
    Call this on some test images y_in, to get a print-out of
    the layer sizes. Shapes shown are (batchsize,pixels,pixels,channels).
    After a call to the visualization routine, y_target will contain
    the last set of training images, so you could feed those in here.
    """
    layer_features=get_layer_activations(network,y_in)
    for idx,feature in enumerate(layer_features):
        s=np.shape(feature)
        print("Layer "+str(idx)+": "+str(s[1]*s[2]*s[3])+" neurons / ", s)

def get_layer_activation_extractor(network):
    return(Model(inputs=network.inputs,
                            outputs=[layer.output for layer in network.layers]))

def get_layer_activations(network, y_in):
    """
    Call this on some test images y_in, to get the intermediate 
    layer neuron values. These are returned in a list, with one
    entry for each layer (the entries are arrays).
    """
    extractor=get_layer_activation_extractor(network)
    layer_features = extractor(y_in)
    return(layer_features)


# ## Example 1: Reproducing randomly placed circles
# 
# This is not even an autoencoder: it never shrinks the size of the layers, so in principle it should eventually work perfectly, but it still has to be trained!

# In[182]:


def my_generator(batchsize,x,y):
    R=np.random.uniform(size=batchsize)
    x0=np.random.uniform(size=batchsize,low=-1,high=1)
    y0=np.random.uniform(size=batchsize,low=-1,high=1)
    return( 1.0*((x[None,:,:]-x0[:,None,None])**2 + (y[None,:,:]-y0[:,None,None])**2 < R[:,None,None]**2) )

Net=Sequential()
# 3x3 kernel size, 10 channels in first hidden layer:
Net.add(Conv2D(10,3,input_shape=(None,None,1),
               activation="sigmoid",padding='same'))
# 3x3 kernel size, only 1 channel in last hidden layer:
Net.add(Conv2D(1,3,activation="linear",padding='same'))
Net.compile(loss='mean_squared_error',
              optimizer='adam')


# In[185]:


visualize_CNN_training(Net, my_generator, 50,
                    steps=100, batchsize=10,
                              visualize_nsteps=10,
                      plot_img_pixels=50, plot_img_rows=3, plot_img_cols=5)


# In[96]:


# show the typical training images (these are available
# in the global variable y_target after calling the visualization routine)
fig,ax=plt.subplots(ncols=10,nrows=1,figsize=(10,1))
for j in range(10):
    ax[j].imshow(y_target[j,:,:,0],origin='lower') # the last training images...
    ax[j].axis('off')
plt.show()


# ## Example 2: Reproducing randomly placed circles with a true autoencoder
# 

# In[186]:


def my_generator(batchsize,x,y):
    R=np.random.uniform(size=batchsize)
    x0=np.random.uniform(size=batchsize,low=-1,high=1)
    y0=np.random.uniform(size=batchsize,low=-1,high=1)
    return( 1.0*((x[None,:,:]-x0[:,None,None])**2 + (y[None,:,:]-y0[:,None,None])**2 < R[:,None,None]**2) )

Net=Sequential()
# 3x3 kernel size, 10 channels in first hidden layer:
Net.add(Conv2D(4,5,input_shape=(None,None,1),
               activation="sigmoid",padding='same'))
Net.add(AveragePooling2D(pool_size=(3,3),padding='same')) # down
Net.add(Conv2D(4,5,
               activation="sigmoid",padding='same'))
Net.add(AveragePooling2D(pool_size=(3,3),padding='same')) # down
Net.add(Conv2D(1,3,
               activation="sigmoid",padding='same'))
Net.add(UpSampling2D(size=(3,3))) # up
Net.add(Conv2D(4,5,
               activation="sigmoid",padding='same'))
Net.add(UpSampling2D(size=(3,3))) # up
Net.add(Conv2D(4,5,
               activation="sigmoid",padding='same'))
Net.add(Conv2D(1,3,activation="linear",padding='same'))
Net.compile(loss='mean_squared_error',
              optimizer='adam')


# In[179]:


Net.summary()


# In[191]:


visualize_CNN_training(Net, my_generator, 9*3,
                    steps=500, batchsize=30,
                              visualize_nsteps=10,
                      plot_img_cols=8, plot_img_rows=4,
                      plot_img_pixels=27)


# In[140]:


print_layers(Net,y_target) # find out layer sizes for these test images!
# y_target is a global variable that is initialized by the
# training visualization routine, and it contains the last few training images.


# # Tutorial Exercise 1: Try "relu" instead of sigmoid!
# 
# How does the appearance of the pictures change? Inspect and try to interpret the intermediate layer results shown in the upper left part of the figure.

# # Tutorial Exercise 2: Try this on another type of pictures!
# 
# Change the my_generator accordingly.
# 

# # Homework Problem: The Grand Autoencoder Challenge
# 
# Using the above image generator (which produces random circles), and the above image size 27x27, try to set up an autoencoder network that 'works best'. The rules of the game are:
# 
# - You may create any kind of network, but the narrowest layer (the bottleneck) must only contain no more than 3 neurons (this defines the 'HARD' version of the challenge). Alternatively, it must only contain no more than 9 neurons (the 'MEDIUM' version of the challenge). When counting neurons, use the routine print_layers (see above!) after running a bit of the training.
# 
# - You may also use any kind of optimizer, any settings for the learning rate, and any choice of batch size. No pre-processing/post-processing of the input/output of the network is allowed.
# 
# - The performance will be judged in the following way: when you start fresh training (after freshly initializing the network), what is the cost value after the network has been trained on 30000 images? (e.g. batchsize=30 and steps=1000). Look at the cost value that is printed after the training run is completed. This must be reproducible, i.e. in case of doubt or if there are several contestants with close cost values, the networks will be trained multiple times, to obtain more precise average cost values.
# 
# - There is also the LONG-TRAINING version of the challenge: what is the cost value you can reach after training over 100,000 images?
# 
# To be competitive for the MEDIUM challenge, your final cost should be below 0.02.
# 
# Please post your intermediate achieved cost functions on the forum, so others can get inspired to fine-tune their network structure!
# 
