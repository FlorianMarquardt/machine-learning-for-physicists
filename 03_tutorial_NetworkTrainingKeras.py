
# coding: utf-8

# # Visualizing the training of Neural Networks with Keras

# Example code for the lecture series "Machine Learning for Physicists" by Florian Marquardt
# 
# Lecture 3, Tutorials
# 
# See https://machine-learning-for-physicists.org and the current course website linked there!

# This notebook shows how to:
# - visualize the training of neural networks using keras
# 
# The networks have 2 input and 1 output neurons, but arbitrarily many hidden layers, and also you can choose the activation functions
# 
# This is essentially an extension of the lecture-2 training notebook, but now using keras instead of pure python.

# ### Imports: numpy and matplotlib and keras

# In[1]:


# keras: Sequential is the neural-network class, Dense is
# the standard network layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
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

# In[2]:


# backpropagation and training routines
# these are now just front-ends to keras!

def apply_net(y_in): # one forward pass through the network
    global Net    
    return(Net.predict_on_batch(y_in))

def apply_net_simple(y_in): # one forward pass through the network
    return(apply_net(y_in))
        
def train_net(y_in,y_target): # one full training batch
    # y_in is an array of size batchsize x (input-layer-size)
    # y_target is an array of size batchsize x (output-layer-size)
    global Net
    
    cost=Net.train_on_batch(y_in,y_target)[0]
    return(cost)

def init_layer_variables(weights,biases,activations,use_keras_init=False,eta=0.1,optimizer='sgd'):
    global Weights, Biases, NumLayers, Activations
    global LayerSizes, y_layer, df_layer, dw_layer, db_layer
    global Net
    
    # store the main data in global variables
    Weights=weights
    Biases=biases
    Activations=activations
    NumLayers=len(Weights)

    # keras activation names can be slightly different from what I used...
    # also: 'jump' is not implemented here
    KerasActivation={ "sigmoid":"sigmoid", "reLU":"relu", "linear":"linear" }

    Net=Sequential() # a new network ('sequential' is the simplest structure: layer by layer)
    
    # now build up the network, layer by layer:
    LayerSizes=[2]
    for j in range(NumLayers):
        LayerSizes.append(len(Biases[j]))
        if use_keras_init: # use keras' random weight initialization approach
            Net.add(Dense(len(Biases[j]), # number of neurons
                          input_shape=(LayerSizes[j],), # size of previous layer
                          activation=KerasActivation[activations[j]] # activation function
                     ))            
        else:
            Net.add(Dense(len(Biases[j]), # number of neurons
                          input_shape=(LayerSizes[j],), # size of previous layer
                          activation=KerasActivation[activations[j]], # activation function
                         weights = [ np.array(weights[j]), np.array(biases[j]) ] # the weights and biases for this layer
                     ))
    if optimizer=='adam':
        the_optimizer=optimizers.Adam(lr=eta) # adaptive
    else:
        the_optimizer=optimizers.SGD(lr=eta) # standard gradient descent with given learning rate!
    Net.compile(loss='mean_squared_error',
              optimizer=the_optimizer,
              metrics=['accuracy'])
    


# In[3]:


# visualization routines:

# some internal routines for plotting the network:
def plot_connection_line(ax,X,Y,W,vmax=1.0,linewidth=3):
    t=np.linspace(0,1,20)
    if W>0:
        col=[0,0.4,0.8]
    else:
        col=[1,0.3,0]
    ax.plot(X[0]+(3*t**2-2*t**3)*(X[1]-X[0]),Y[0]+t*(Y[1]-Y[0]),
           alpha=abs(W)/vmax,color=col,
           linewidth=linewidth)
    
def plot_neuron_alpha(ax,X,Y,B,size=100.0,vmax=1.0):
    if B>0:
        col=np.array([[0,0.4,0.8]])
    else:
        col=np.array([[1,0.3,0]])
    ax.scatter([X],[Y],marker='o',c=col,alpha=abs(B)/vmax,s=size,zorder=10)

def plot_neuron(ax,X,Y,B,size=100.0,vmax=1.0):
    if B>0:
        col=np.array([[0,0.4,0.8]])
    else:
        col=np.array([[1,0.3,0]])
    ax.scatter([X],[Y],marker='o',c=col,s=size,zorder=10)
    
def visualize_network(weights,biases,activations,
                      M=100,y0range=[-1,1],y1range=[-1,1],
                     size=400.0, linewidth=5.0,
                     weights_are_swapped=False,
                    layers_already_initialized=False,
                      plot_cost_function=None,
                      current_cost=None, cost_max=None, plot_target=None
                     ):
    """
    Visualize a neural network with 2 input 
    neurons and 1 output neuron (plot output vs input in a 2D plot)
    
    weights is a list of the weight matrices for the
    layers, where weights[j] is the matrix for the connections
    from layer j to layer j+1 (where j==0 is the input)
    
    weights[j][m,k] is the weight for input neuron k going to output neuron m
    (note: internally, m and k are swapped, see the explanation of
    batch processing in lecture 2)
    
    biases[j] is the vector of bias values for obtaining the neurons in layer j+1
    biases[j][k] is the bias for neuron k in layer j+1
    
    activations is a list of the activation functions for
    the different layers: choose 'linear','sigmoid',
    'jump' (i.e. step-function), and 'reLU'
    
    M is the resolution (MxM grid)
    
    y0range is the range of y0 neuron values (horizontal axis)
    y1range is the range of y1 neuron values (vertical axis)
    """
    if not weights_are_swapped:
        swapped_weights=[]
        for j in range(len(weights)):
            swapped_weights.append(np.transpose(weights[j]))
    else:
        swapped_weights=weights

    y0,y1=np.meshgrid(np.linspace(y0range[0],y0range[1],M),np.linspace(y1range[0],y1range[1],M))
    y_in=np.zeros([M*M,2])
    y_in[:,0]=y0.flatten()
    y_in[:,1]=y1.flatten()
    
    # if we call visualization directly, we still
    # need to initialize the 'Weights' and other
    # global variables; otherwise (during training)
    # all of this has already been taken care of:
    if not layers_already_initialized:
        init_layer_variables(swapped_weights,biases,activations)
    y_out=apply_net_simple(y_in)

    if plot_cost_function is None:
        fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(8,4))
    else:
        fig=plt.figure(figsize=(8,4))
        gs_top = gridspec.GridSpec(nrows=1, ncols=2)
        gs_left = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_top[0], height_ratios=[1.0,0.3])
        ax=[ fig.add_subplot(gs_left[0]),
            fig.add_subplot(gs_top[1]),
           fig.add_subplot(gs_left[1]) ]
        # ax[0] is network
        # ax[1] is image produced by network
        # ax[2] is cost function subplot
        
    # plot the network itself:
    
    # positions of neurons on plot:
    posX=[[-0.5,+0.5]]; posY=[[0,0]]
    vmax=0.0 # for finding the maximum weight
    vmaxB=0.0 # for maximum bias
    for j in range(len(biases)):
        n_neurons=len(biases[j])
        posX.append(np.array(range(n_neurons))-0.5*(n_neurons-1))
        posY.append(np.full(n_neurons,j+1))
        vmax=np.maximum(vmax,np.max(np.abs(weights[j])))
        vmaxB=np.maximum(vmaxB,np.max(np.abs(biases[j])))

    # plot connections
    for j in range(len(biases)):
        for k in range(len(posX[j])):
            for m in range(len(posX[j+1])):
                plot_connection_line(ax[0],[posX[j][k],posX[j+1][m]],
                                     [posY[j][k],posY[j+1][m]],
                                     swapped_weights[j][k,m],vmax=vmax,
                                    linewidth=linewidth)
    
    # plot neurons
    for k in range(len(posX[0])): # input neurons (have no bias!)
        plot_neuron(ax[0],posX[0][k],posY[0][k],
                   vmaxB,vmax=vmaxB,size=size)
    for j in range(len(biases)): # all other neurons
        for k in range(len(posX[j+1])):
            plot_neuron(ax[0],posX[j+1][k],posY[j+1][k],
                       biases[j][k],vmax=vmaxB,size=size)
    
    ax[0].axis('off')
    
    # now: the output of the network
    img=ax[1].imshow(np.reshape(y_out,[M,M]),origin='lower',
                    extent=[y0range[0],y0range[1],y1range[0],y1range[1]])
    ax[1].set_xlabel(r'$y_0$')
    ax[1].set_ylabel(r'$y_1$')
    
#     axins1 = inset_axes(ax[1],
#                     width="40%",  # width = 50% of parent_bbox width
#                     height="5%",  # height : 5%
#                     loc='upper right',
#                        bbox_to_anchor=[0.3,0.4])

#    axins1 = ax[1].inset_axes([0.5,0.8,0.45,0.1])
    axins1 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax[1], [0.25, 0.1, 0.5, 0.05])
    axins1.set_axes_locator(ip)

    imgmin=np.min(y_out)
    imgmax=np.max(y_out)
    color_bar=fig.colorbar(img, cax=axins1, orientation="horizontal",ticks=np.linspace(imgmin,imgmax,3))
    cbxtick_obj = plt.getp(color_bar.ax.axes, 'xticklabels')
    plt.setp(cbxtick_obj, color="white")
    axins1.xaxis.set_ticks_position("bottom")

    if plot_target is not None:
        axins2 = plt.axes([0.01, 0.01, 0.99, 0.99])
        ip = InsetPosition(ax[1], [0.75, 0.75, 0.2, 0.2])
        axins2.set_axes_locator(ip)
        axins2.imshow(plot_target,origin='lower')
        axins2.get_xaxis().set_ticks([])
        axins2.get_yaxis().set_ticks([])
        
    if plot_cost_function is not None:
        ax[2].plot(plot_cost_function)
        ax[2].set_ylim([0.0,cost_max])
        ax[2].set_yticks([0.0,cost_max])
        ax[2].set_yticklabels(["0",'{:1.2e}'.format(cost_max)])
        if current_cost is not None:
            ax[2].text(0.9, 0.9, 'cost={:1.2e}'.format(current_cost), horizontalalignment='right',
                       verticalalignment='top', transform=ax[2].transAxes)
    
    plt.show()
    
def visualize_network_training(weights,biases,activations,
                               target_function,
                               num_neurons=None,
                               weight_scale=1.0,
                               bias_scale=1.0,
                               yspread=1.0,
                      M=100,y0range=[-1,1],y1range=[-1,1],
                     size=400.0, linewidth=5.0,
                    steps=100, batchsize=10, eta=0.1,
                              random_init=False,
                              visualize_nsteps=1,
                              plot_target=True,
                              use_keras_init=False,
                              optimizer='sgd'):
    """
    Visualize the training of a neural network.
    
    weights, biases, and activations define the neural network 
    (the starting point of the optimization; for the detailed description,
    see the help for visualize_network)
    
    If you want to have layers randomly initialized, just provide
    the number of neurons for each layer as 'num_neurons'. This should include
    all layers, including input (2 neurons) and output (1), so num_neurons=[2,3,5,4,1] is
    a valid example. In this case, weight_scale and bias_scale define the
    spread of the random Gaussian variables used to initialize all weights and biases.
    
    target_function is the name of the function that we
    want to approximate; it must be possible to 
    evaluate this function on a batch of samples, by
    calling target_function(y) on an array y of 
    shape [batchsize,2], where
    the second index refers to the two coordinates
    (input neuron values) y0 and y1. The return
    value must be an array with one index, corresponding
    to the batchsize. A valid example is:
    
    def my_target(y):
        return( np.sin(y[:,0]) + np.cos(y[:,1]) )
    
    steps is the number of training steps
    
    batchsize is the number of samples per training step
    
    eta is the learning rate (stepsize in the gradient descent)
    
    yspread denotes the spread of the Gaussian
    used to sample points in (y0,y1)-space
    
    visualize_n_steps>1 means skip some steps before
    visualizing again (can speed up things)
    
    plot_target=True means do plot the target function in a corner
    
    For all the other parameters, see the help for
        visualize_network
    
    weights and biases as given here will be used
    as starting points, unless you specify
    random_init=True, in which case they will be
    used to determine the spread of Gaussian random
    variables used for initialization!
    """
    global Net, Weights, Biases
    
    if num_neurons is not None: # build weight matrices as randomly initialized
        weights=[weight_scale*np.random.randn(num_neurons[j+1],num_neurons[j]) for j in range(len(num_neurons)-1)]
        biases=[bias_scale*np.random.randn(num_neurons[j+1]) for j in range(len(num_neurons)-1)]
    
    swapped_weights=[]
    for j in range(len(weights)):
        swapped_weights.append(np.transpose(weights[j]))
    init_layer_variables(swapped_weights,biases,activations,use_keras_init=use_keras_init,eta=eta,optimizer=optimizer)
    
    if plot_target:
        y0,y1=np.meshgrid(np.linspace(y0range[0],y0range[1],M),np.linspace(y1range[0],y1range[1],M))
        y=np.zeros([M*M,2])
        y[:,0]=y0.flatten()
        y[:,1]=y1.flatten()
        plot_target_values=np.reshape(target_function(y),[M,M])
    else:
        plot_target_values=None
    
    y_target=np.zeros([batchsize,1])
    costs=np.zeros(steps)
    
    for j in range(steps):
        # produce samples (random points in y0,y1-space):
        y_in=yspread*np.random.randn(batchsize,2)
        # apply target function to those points:
        y_target[:,0]=target_function(y_in)
        # do one training step on this batch of samples:
        costs[j]=train_net(y_in,y_target)
        
        # now visualize the updated network:
        if j%visualize_nsteps==0:
            clear_output(wait=True) # for animation
            if j>10:
                cost_max=np.average(costs[0:j])*1.5
            else:
                cost_max=costs[0]

            # extract weights and biases from the keras network:
            for j in range(NumLayers):
                Weights[j],Biases[j]=Net.layers[j].get_weights()
            
            visualize_network(Weights,Biases,activations,
                          M,y0range=y0range,y1range=y1range,
                         size=size, linewidth=linewidth,
                             weights_are_swapped=True,
                             layers_already_initialized=True,
                             plot_cost_function=costs,
                             current_cost=costs[j],
                             cost_max=cost_max,
                             plot_target=plot_target_values)
            sleep(0.1) # wait a bit before next step (probably not needed)


# ## Example 1: Training for a simple AND function

# In[4]:


def my_target(y):
    return( 1.0*( (y[:,0]+y[:,1])>0) )

visualize_network_training(weights=[ [ 
    [0.2,-0.9]  # weights of 2 input neurons for single output neuron
    ] ],
    biases=[ 
        [0.0] # bias for single output neuron
            ],
    target_function=my_target, # the target function to approximate
    activations=[ 'sigmoid' # activation for output
                ],
    y0range=[-3,3],y1range=[-3,3],
    steps=1000, eta=.5, batchsize=200,
                          visualize_nsteps=10, 
                           plot_target=True,
                          optimizer='adam')


# ## Example 2: Training for half a smiley (circle with two eyes)

# In[67]:


def my_target(y):
    a=0.8; r=0.5; R=2.0
    return( 1.0*( y[:,0]**2+y[:,1]**2<R**2 ) - 1.0*( (y[:,0]-a)**2+(y[:,1]-a)**2<r**2) - 1.0*( (y[:,0]+a)**2+(y[:,1]-a)**2<r**2 ) )


visualize_network_training(weights=[],biases=[],num_neurons=[2,30,30,1],
    target_function=my_target, # the target function to approximate
    activations=[ 'reLU','reLU','linear' ],
    y0range=[-3,3],y1range=[-3,3],
    steps=5000, eta=.1, batchsize=200,
                          visualize_nsteps=100, 
                           plot_target=True,
                          optimizer='adam',
                          size=20,linewidth=2)


# ## Exercise: Extend this code so as to be able to use more advanced keras activation functions for the layers!
