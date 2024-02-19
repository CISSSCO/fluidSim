#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from time import time as ti
from matplotlib import cm
from tqdm import tqdm
from numba import cuda
import cupy as cp

# # for pytorch
# from torch import zeros, cp.array, roll, sin, sqrt, linspace
# from torch import sum as tsum
# import torch


# In[ ]:


try:
    from cupy.cuda.nvtx import RangePush as nvtxRangePush
    from cupy.cuda.nvtx import RangePop  as nvtxRangePop
except:
    pass

try:
    from nvtx import range_push as nvtxRangePush
    from nvtx import range_pop  as nvtxRangePop
except:
    pass

# from cupy.cuda.nvtx import RangePush as nvtxRangePush
# from cupy.cuda.nvtx import RangePop  as nvtxRangePop


# In[2]:


# print("Torch version:",torch.__version__)

# torch.cuda.get_device_name(0)


# In[3]:


# selecting device as cuda if available otherwise will set to cpu
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


# In[4]:


# @cuda.jit
nvtxRangePush("macroscopic function")
def macroscopic(fin, nx, ny, v):
    rho = cp.sum(fin,axis=0)    # sending data to device
    # rho_gpu = cuda.to_device(rho)
    u = cp.zeros((2,nx,ny))  # sending data to device
    # u_gpu = cuda.to_device(u)
    # i = cuda.grid(1)
    for i in range(9):
        u[0,:,:] += v[i,0]*fin[i,:,:]
        u[1,:,:] += v[i,1]*fin[i,:,:]
    u /= rho
    return rho, u
nvtxRangePop()


# In[5]:


# @cuda.jit
nvtxRangePush("equilibrium function")
def equilibrium(rho, u, v, t, nx, ny):
    usqr = (3/2)*(u[0]**2+u[1]**2)
    feq = cp.zeros((9,nx,ny))
    # i = cuda.grid(1)
    for i in range(9):
        cu = 3*(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*t[i]*(1+cu+0.5*cu**2-usqr)
    return feq
nvtxRangePop()


# In[6]:


nvtxRangePush("obstacle function")
def obstacle_fun(cx, cy, r):
    def inner(x, y):
        return (x-cx)**2+(y-cy)**2<r**2
    return inner
nvtxRangePop()


# In[7]:


nvtxRangePush("inivel function")
def inivel(uLB, ly, d, nx, ny):
    _,yy = cp.meshgrid(cp.linspace(0, nx - 1, nx), cp.linspace(0, ny - 1, ny))
    yy = yy.T
    # yy.to(device)
    # vel = zeros((d, nx, ny)).to(device)
    vel = cp.zeros((d, nx, ny))
    for dir in range(d):
        vel[dir,:,:] = (1-dir) * uLB * (1+1e-4*cp.sin(yy/ly*2*cp.pi))
    return vel
nvtxRangePop()
# def inivel( uLB, ly):
#     def inner(d,x,y):
#         return (1-d) * uLB * (1+1e-4*cp.sin(y/ly*2*np.pi))
#     return inner
    


# In[8]:


Re = 170.0                  # Reynolds number
#------------------------------------------------------------------------------
maxIter = 50000
nx,ny = 680,240             # Domain dimensions
ly = ny-1
uLB = 0.04                  # Inlet velocity NON PHYSICAL??
cx,cy,r = nx//4,ny//2,ny/9  # cylinder coordinates and radius (as integers)
nulb = uLB*r/Re             # Viscosity
omega = 1 / (3*nulb+0.5)    # Relaxation parameter


# In[9]:


# lattice velocities
v = cp.array([
            [1,1],
            [1,0],
            [1,-1],
            [0,1],
            [0,0],
            [0,-1],
            [-1,1],
            [-1,0],
            [-1,-1]
            ])

# weights
t = cp.array([
            1/36,
            1/9,
            1/36,
            1/9,
            4/9,
            1/9,
            1/36,
            1/9,
            1/36
            ])


# In[10]:


col_0 = cp.array([0,1,2])
col_1 = cp.array([3,4,5])
col_2 = cp.array([6,7,8])



# In[11]:


# kwargs['shape'] = [nx, ny]
# instantiate the cylindrical obstacle
obstacle = cp.array(cp.fromfunction(obstacle_fun(cx,cy,r),(nx, ny)))
ob = obstacle.get()
if True:
  plt.imshow(ob)


# In[12]:


# initial velocity profile
# vel = cp.fromfunction(inivel(uLB,ly),(2,nx,ny))
vel = inivel(uLB, ly, 2, nx, ny)
# tpb = 16
# size = 10000
# block = (size // tpb)
# initialize fin to equilibirum (rho = 1)
# fin = equilibrium[block, tpb](1,vel,v,t,nx,ny)
fin = equilibrium(1,vel,v,t,nx,ny)

#==============================================================================
#   Time-Stepping
#==============================================================================
t0 = ti()
for time in tqdm(range(maxIter)):
    # outflow boundary condition (right side) NEUMANN BC! No gradient
    fin[col_2,-1,:] = fin[col_2,-2,:]
    # compute macroscopic variables
    
    # rho,u = macroscopic[block, tpb](fin,nx,ny,v)
    rho,u = macroscopic(fin,nx,ny,v)

    # inlet boundary condition (left wall)
    u[:,0,:] = vel[:,0,:]
    rho[0,:] = 1/(1-u[0,0,:])*( cp.sum(fin[col_1,0,:], axis = 0)+
                                2*cp.sum(fin[col_2,0,:], axis = 0))

    feq = equilibrium(rho,u,v,t,nx,ny)
    fin[col_0,0,:] = feq[col_0,0,:] + fin[col_2,0,:]-feq[col_2,0,:]

    # Collide
    fout = fin - omega*(fin-feq)

    # bounceback
    for i in range(9):
        fout[i,obstacle] = fin[8-i,obstacle]

    # stream
    for i in range(9):
        # be careful with this -> numpy.roll cycles through an array by an axis
        # and the last element becomes the first. this implements a periodic
        # boundary in a very compact syntax, but needs to be reworked for other
        # implementations
        fin[i,:,:] = cp.roll(
                          cp.roll(
                                fout[i,:,:], v[i,0], axis = 0
                               ),
                          v[i,1], axis = 1
                          )

    # Output an image every 100 iterations
    if (time%100 == 0):
        plt.clf()
        u_temp = u.get()
        # x_temp = int(round(5 * nx / ny))
        # y_temp = int(round(5))
        plt.imshow(np.sqrt(u_temp[0]**2+u_temp[1]**2).T, cmap= 'jet')
        plt.title(f"Velocity Magnitude at Iteration {time}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("./cupyTest/vel{0:03d}.png".format(time//100))
tf = ti() - t0

print("time to execute = ",tf)


# In[ ]:





# In[ ]:




