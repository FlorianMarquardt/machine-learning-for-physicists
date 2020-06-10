
# coding: utf-8

# # Neural Networks with Pure Python

# Example code for the lecture series "Machine Learning for Physicists" by Florian Marquardt
# 
# Lecture 1
# 
# See https://machine-learning-for-physicists.org and the current course website linked there!

# This notebook shows how to:
# - implement the forward-pass (evaluation) of a deep, fully connected neural network in a few lines of python
# - do that efficiently using batches
# - illustrate the results for randomly initialized neural networks

# ### Imports: only numpy and matplotlib

# In[1]:


# get the "numpy" library for linear algebra

# In the lecture videos, I do this:
#
# from numpy import *
#
# WARNING: It is generally considered bad
# programming style to "import *", as it
# can lead to confusion. For me, I
#  (1) ALWAYS import numpy
#  (2) NEVER import any other package in this * way
# Therefore, there is never confusion for me, and
# it makes my code a bit more readable (for me).
# However, since 99% of people are using the 
# syntax "import numpy as np" and then
# access "np.exp()" etc., you
# should probably also use "np" once you start
# exchanging code with others. I convert
# back to the np. syntax when I turn my
# converged code into a module.
#
# It is apparently officially accepted to explicitly
# list all the functions you need from numpy:

from numpy import array, zeros, exp, random, dot, shape, reshape, meshgrid, linspace

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display


# ## A very simple neural network (no hidden layer)

# A network with N0 input neurons and N1 output neurons (no hidden layer)
# 
# $$y^{\rm out}_j = f(\sum_k w_{jk} y^{\rm in}_k + b_k)$$
# 
# where $w$ is the weight matrix, $b$ is the bias vector, and $f$ would be the activation function (e.g. the sigmoid here), which is applied independently for each $j$.

# In[2]:


N0=3 # input layer size
N1=2 # output layer size

# initialize random weights: array dimensions N1xN0
w=random.uniform(low=-1,high=+1,size=(N1,N0))
# initialize random biases: N1 vector
b=random.uniform(low=-1,high=+1,size=N1)


# In[3]:


# input values
y_in=array([0.2,0.4,-0.1])


# In[4]:


# evaluate network by hand, in two steps
z=dot(w,y_in)+b # result: the vector of 'z' values, length N1
y_out=1/(1+exp(-z)) # the 'sigmoid' function (applied elementwise)


# In[5]:


print("network input y_in:", y_in)
print("weights w:", w)
print("bias vector b:", b)
print("linear superposition z:", z)
print("network output y_out:", y_out)


# ### Visualize network result

# Still stay with the simple network, but define a function that evaluates the network, and visualize the  output for various inputs

# In[6]:


# a function that applies the network
def apply_net(y_in):
    global w, b
    
    z=dot(w,y_in)+b    
    return(1/(1+exp(-z)))


# In[7]:


N0=2 # input layer size
N1=1 # output layer size

w=random.uniform(low=-10,high=+10,size=(N1,N0)) # random weights: N1xN0
b=random.uniform(low=-1,high=+1,size=N1) # biases: N1 vector


# In[8]:


apply_net([0.8,0.3]) # a simple test


# In[9]:


# Note: this is NOT the most efficient way to do this! (but simple)
# We will later learn to use array syntax efficiently

M=50 # will create picture of size MxM
y_out=zeros([M,M]) # array MxM, to hold the result

for j1 in range(M):
    for j2 in range(M):
        # out of these integer indices, generate
        # two values in the range -0.5...0.5
        # and then apply the network to those two
        # input values
        value0=float(j1)/M-0.5
        value1=float(j2)/M-0.5
        y_out[j1,j2]=apply_net([value0,value1])[0]


# In[10]:


# display image
plt.imshow(y_out,origin='lower',extent=(-0.5,0.5,-0.5,0.5))
plt.colorbar()
plt.title("NN output as a function of input values")
plt.xlabel("y_2")
plt.ylabel("y_1")
plt.show()


# ### Now visualize a network with one hidden layer

# The idea here is to have multiple weight matrices (for each pair of subsequent layers there is one weight matrix). The function that "applies a layer", i.e. goes from one layer to the next, is essentially the same as the function evaluating the simple network above. 

# In[11]:


# a function that evaluates one layer based
# on the neuron values in the preceding layer
def apply_layer(y_in,w,b): 
    z=dot(w,y_in)+b
    return(1/(1+exp(-z)))


# In[12]:


N0=2 # input layer size
N1=30 # hidden layer size
N2=1 # output layer size

# weights and biases
# from input layer to hidden layer:
w1=random.uniform(low=-10,high=+10,size=(N1,N0)) # random weights: N1xN0
b1=random.uniform(low=-1,high=+1,size=N1) # biases: N1 vector

# weights+biases from hidden layer to output layer:
w2=random.uniform(low=-10,high=+10,size=(N2,N1)) # random weights
b2=random.uniform(low=-1,high=+1,size=N2) # biases


# In[13]:


# evaluate the network by subsequently
# evaluating the two steps (input to hidden and
# hidden to output)
def apply_net(y_in):
    global w1,b1,w2,b2
    
    y1=apply_layer(y_in,w1,b1)
    y2=apply_layer(y1,w2,b2)
    return(y2)


# In[14]:


# Again, obtain values for a range of inputs
# Note: this is NOT the most efficient way to do this! (but simple)

M=50 # will create picture of size MxM
y_out=zeros([M,M]) # array MxM, to hold the result

for j1 in range(M):
    for j2 in range(M):
        value0=float(j1)/M-0.5
        value1=float(j2)/M-0.5
        y_out[j1,j2]=apply_net([value0,value1])[0]


# In[15]:


# display image
plt.imshow(y_out,origin='lower',extent=(-0.5,0.5,-0.5,0.5))
plt.colorbar()
plt.title("NN output as a function of input values")
plt.xlabel("y_2")
plt.ylabel("y_1")
plt.show()


# Obviously, the shape of the output is already more 'complex' than that of a simple network without hidden layer! Let's go further in that direction...

# ### Applying a network to a 'batch' of samples (python trickery)

# Goal: apply network to many samples in parallel (no 'for' loops!)

# #### Small excursion: matrix-vector multiplication and python broadcasting of array dimensions

# In[16]:


# See how the dot product works: 
# 'contracts' (sums over) the innermost index
W=zeros([7,8])
y=zeros([8,30]) 
# here '30' would stand for the number of samples
# in our envisaged network applications
shape(dot(W,y))


# In[17]:


# now try to add the bias vector entries,
# in the most naive way (beware!)
B=zeros(7)
result=dot(W,y)+B # will produce an error!


# In[18]:


# But with a re-ordering of indices, this works!
# So, let's take the dimension of size 30 to be
# the very first one:
y=zeros([30,8])
W=zeros([8,7])
shape(dot(y,W))


# In[19]:


# and now try again adding the bias vector,
# again in a naive way
B=zeros(7)
result=dot(y,W)+B 
# will give the desired result, 
# because B is 'broadcast' to shape (30,7)
shape(result)


# #### Defining the functions that evaluate a layer and evaluate the network, with batch processing

# Set up for batch processing, i.e. parallel evaluation of many input samples!

# In[20]:


def apply_layer_new(y_in,w,b): # a function that applies a layer    
    z=dot(y_in,w)+b # note different order in matrix product!
    return(1/(1+exp(-z)))


# In[21]:


def apply_net_new(y_in): # same as before, but with new layer function
    global w1,b1,w2,b2
    
    y1=apply_layer_new(y_in,w1,b1)
    y2=apply_layer_new(y1,w2,b2)
    return(y2)


# In[22]:


N0=2 # input layer size
N1=5 # hidden layer size
N2=1 # output layer size

# from input layer to hidden layer:
w1=random.uniform(low=-10,high=+10,size=(N0,N1)) # NEW ORDER!! N0xN1
b1=random.uniform(low=-1,high=+1,size=N1) # biases: N1 vector

# from hidden layer to output layer:
w2=random.uniform(low=-10,high=+10,size=(N1,N2)) # NEW ORDER N1xN2
b2=random.uniform(low=-1,high=+1,size=N2) # biases


# In[23]:


batchsize=10000
y=random.uniform(low=-1,high=1,size=(batchsize,2))


# In[24]:


y_out=apply_net_new(y)


# In[25]:


shape(y_out) 
# these were 10000 samples evaluated in parallel!!!


# ### Now visualize multi-layer net again, but more efficiently!

# All the pixels in the image are samples, process all of them together!

# In[26]:


M=50
# Generate a 'mesh grid', i.e. x,y values in an image
v0,v1=meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))


# In[27]:


fig,ax=plt.subplots(1,2)
ax[0].imshow(v0,origin='lower')
ax[0].set_title("input value v0")
ax[1].imshow(v1,origin='lower')
ax[1].set_title("input value v1")
plt.show()


# In[28]:


v0flat=v0.flatten() # make 1D array out of 2D array
v1flat=v1.flatten() 
# that means: MxM matrix becomes M^2 vector
shape(v0flat)


# In[29]:


batchsize=shape(v0flat)[0] # number of samples = number of pixels
y_in=zeros([batchsize,2])
y_in[:,0]=v0flat # fill first component (index 0)
y_in[:,1]=v1flat # fill second component


# In[30]:


# apply net to all these samples simultaneously!
y_out=apply_net_new(y_in) 


# In[31]:


shape(y_out) # this is not a vector but a funny matrix batchsize x 1


# In[32]:


# turn this back into a 2D matrix (image)
y_2D=reshape(y_out[:,0],[M,M]) 


# In[33]:


plt.imshow(y_2D,origin='lower')
plt.title("NN output (one hidden layer)")
plt.xlabel("v0")
plt.ylabel("v1")
plt.show()


# ### For fun: a network with MANY hidden layers

# In[34]:


Nlayers=20 # not counting the input layer & the output layer
LayerSize=100

Weights=random.uniform(low=-3,high=3,size=[Nlayers,LayerSize,LayerSize])
Biases=random.uniform(low=-1,high=1,size=[Nlayers,LayerSize])

# for the first hidden layer (coming in from the input layer)
WeightsFirst=random.uniform(low=-1,high=1,size=[2,LayerSize])
BiasesFirst=random.uniform(low=-1,high=1,size=LayerSize)

# for the final layer (i.e. the output neuron)
WeightsFinal=random.uniform(low=-1,high=1,size=[LayerSize,1])
BiasesFinal=random.uniform(low=-1,high=1,size=1)


# In[35]:


def apply_multi_net(y_in):
    global Weights, Biases, WeightsFinal, BiasesFinal, Nlayers
    
    y=apply_layer_new(y_in,WeightsFirst,BiasesFirst)    
    for j in range(Nlayers):
        y=apply_layer_new(y,Weights[j,:,:],Biases[j,:])
    output=apply_layer_new(y,WeightsFinal,BiasesFinal)
    return(output)


# In[36]:


# Generate a 'mesh grid', i.e. x,y values in an image
M=40
v0,v1=meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in=zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component


# In[37]:


# use the MxM input grid that we generated above 
y_out=apply_multi_net(y_in) # apply net to all these samples!


# In[38]:


y_2D=reshape(y_out[:,0],[M,M]) # back to 2D image


# In[39]:


plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest')
plt.colorbar()
plt.show()


# Now do the same, but high-resolution (400x400 picture)

# In[40]:


M=400
# Generate a 'mesh grid', i.e. x,y values in an image
v0,v1=meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in=zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component


# The next function takes a few seconds:

# In[41]:


# use the MxM input grid that we generated above 
y_out=apply_multi_net(y_in) # apply net to all these samples!


# In[42]:


y_2D=reshape(y_out[:,0],[M,M]) # back to 2D image


# In[43]:


plt.figure(figsize=[10,10])
plt.axes([0,0,1,1]) # fill all of the picture with the image
plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest')
plt.axis('off') # no axes
plt.show()


# ### Try another nonlinearity: rectified linear units!

# In[44]:


x=linspace(-1,1,100)
plt.plot(x,x*(x>0),linewidth=5)
plt.show()


# In[45]:


def apply_layer_new_relu(y_in,w,b): # a function that applies a layer    
    z=dot(y_in,w)+b # note different order in matrix product!
    return(z*(z>0))


# In[46]:


def apply_multi_net_relu(y_in):
    global Weights, Biases, WeightsFinal, BiasesFinal, Nlayers
    
    y=apply_layer_new_relu(y_in,WeightsFirst,BiasesFirst)    
    for j in range(Nlayers):
        y=apply_layer_new_relu(y,Weights[j,:,:],Biases[j,:])
    output=apply_layer_new_relu(y,WeightsFinal,BiasesFinal)
    return(output)


# In[47]:


Nlayers=20 # not counting the input layer & the output layer
LayerSize=100

Weights=random.uniform(low=-10,high=10,size=[Nlayers,LayerSize,LayerSize])
Biases=random.uniform(low=-1,high=1,size=[Nlayers,LayerSize])

# for the first hidden layer (coming in from the input layer)
WeightsFirst=random.uniform(low=-1,high=1,size=[2,LayerSize])
BiasesFirst=random.uniform(low=-1,high=1,size=LayerSize)

# for the final layer (i.e. the output neuron)
WeightsFinal=random.uniform(low=-1,high=1,size=[LayerSize,1])
BiasesFinal=random.uniform(low=-1,high=1,size=1)


# In[48]:


M=400
# Generate a 'mesh grid', i.e. x,y values in an image
v0,v1=meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in=zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component


# In[49]:


# use the MxM input grid that we generated above 
y_out=apply_multi_net_relu(y_in) # apply net to all these samples!


# In[50]:


y_2D=reshape(y_out[:,0],[M,M]) # back to 2D image


# In[51]:


plt.figure(figsize=[10,10])
plt.axes([0,0,1,1]) # fill all of the picture with the image
plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest')
plt.axis('off') # no axes
plt.show()


# Now zoom out!

# In[52]:


M=400
# Generate a 'mesh grid', i.e. x,y values in an image
v0,v1=meshgrid(linspace(-10,10,M),linspace(-10,10,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in=zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component


# In[53]:


# use the MxM input grid that we generated above 
y_out=apply_multi_net_relu(y_in) # apply net to all these samples!


# In[54]:


y_2D=reshape(y_out[:,0],[M,M]) # back to 2D image


# In[55]:


plt.figure(figsize=[10,10])
plt.axes([0,0,1,1]) # fill all of the picture with the image
plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest')
plt.axis('off') # no axes
plt.show()


# In[56]:


plt.plot(y_2D[:,200])
plt.show()

