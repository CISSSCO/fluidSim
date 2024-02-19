#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from time import time as ti
from tqdm import tqdm

# for pytorch
from torch import zeros, tensor, roll, sin, sqrt, linspace
from torch import sum as tsum
import torch


# In[36]:


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


# In[37]:


#-----------------------------------------------------
# Code for profiling cuda version
#-----------------------------------------------------
# def nvtxRangePush(*arg):
#     pass
# def nvtxRangePop(*arg):
#     pass


# In[38]:


print("Torch version:",torch.__version__)

torch.cuda.get_device_name(0)


# In[39]:


# ! nvcc --version


# In[40]:


# selecting device as cuda if available otherwise will set to cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[41]:


nvtxRangePush("macroscopic function")
def macroscopic(fin, nx, ny, v):
    rho = tsum(fin,axis=0).to(device)   # sending data to device
    u = zeros((2,nx,ny)).to(device) # sending data to device
    for i in range(9):
        u[0,:,:] += v[i,0]*fin[i,:,:]
        u[1,:,:] += v[i,1]*fin[i,:,:]
    u /= rho
    return rho, u
nvtxRangePop()


# In[42]:


nvtxRangePush("obstacle function")
def obstacle_fun(cx, cy, r):
    def inner(x, y):
        return (x-cx)**2+(y-cy)**2<r**2
    return inner
nvtxRangePop()


# In[43]:


nvtxRangePush("equilibrium function")
def equilibrium(rho, u, v, t, nx, ny):
    usqr = (3/2)*(u[0]**2+u[1]**2)
    feq = zeros((9,nx,ny))
    for i in range(9):
        cu = 3*(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*t[i]*(1+cu+0.5*cu**2-usqr)
    return feq.to(device)
nvtxRangePop()


# In[44]:


nvtxRangePush("inivel function")
def inivel(uLB, ly, d, nx, ny):
    _,yy = torch.meshgrid(linspace(0, nx - 1, nx), linspace(0, ny - 1, ny))
    yy.to(device)
    vel = zeros((d, nx, ny)).to(device)
    for dir in range(d):
        vel[dir,:,:] = (1-dir) * uLB * (1+1e-4*sin(yy/ly*2*np.pi))
    return vel
nvtxRangePop()


# In[45]:


Re = 170.0                  # Reynolds number
#------------------------------------------------------------------------------
maxIter = 50000
nx,ny = 680,240             # Domain dimensions
ly = ny-1
uLB = 0.04                  # Inlet velocity NON PHYSICAL??
cx,cy,r = nx//4,ny//2,ny/9  # cylinder coordinates and radius (as integers)
nulb = uLB*r/Re             # Viscosity
omega = 1 / (3*nulb+0.5)    # Relaxation parameter


# In[46]:


# lattice velocities
v = tensor([
            [1,1],
            [1,0],
            [1,-1],
            [0,1],
            [0,0],
            [0,-1],
            [-1,1],
            [-1,0],
            [-1,-1]
            ]).int().to(device)

# weights
t = tensor([
            1/36,
            1/9,
            1/36,
            1/9,
            4/9,
            1/9,
            1/36,
            1/9,
            1/36
            ]).float().to(device)


# In[47]:


col_0 = tensor([0,1,2]).long().to(device)
col_1 = tensor([3,4,5]).long().to(device)
col_2 = tensor([6,7,8]).long().to(device)



# In[48]:


obstacle = torch.tensor(np.fromfunction(obstacle_fun(cx,cy,r),(nx, ny)))
if True:
  plt.imshow(obstacle)


# In[49]:


# initial velocity profile
#vel = np.fromfunction(inivel(uLB,ly),(2,nx,ny))
vel = inivel(uLB, ly, 2, nx, ny)

# initialize fin to equilibirum (rho = 1)
fin = equilibrium(1,vel,v,t,nx,ny).to(device)

#==============================================================================
#   Time-Stepping
#==============================================================================
t0 = ti()
for time in tqdm(range(maxIter)):
    # outflow boundary condition (right side) NEUMANN BC! No gradient
    fin[col_2,-1,:] = fin[col_2,-2,:]

    # compute macroscopic variables
    rho,u = macroscopic(fin,nx,ny,v)

    # inlet boundary condition (left wall)
    u[:,0,:] = vel[:,0,:]
    rho[0,:] = 1/(1-u[0,0,:])*( tsum(fin[col_1,0,:], axis = 0)+
                                2*tsum(fin[col_2,0,:], axis = 0))

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
        fin[i,:,:] = roll(
                          roll(
                                fout[i,:,:], v[i,0].item(), dims = 0
                               ),
                          v[i,1].item(), dims = 1
                          )

    # Output an image every 100 iterations
    if (time%100 == 0):
        plt.clf()
        u_temp = u.cpu()
        # x_temp = int(round(5 * nx / ny))
        # y_temp = int(round(5))
        plt.title(f"Velocity Magnitude at Iteration {time}")
        # plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.imshow(sqrt(u_temp[0]**2+u_temp[1]**2).T, cmap= 'jet')
        plt.savefig("./pytorchTest/vel{0:03d}.png".format(time//100))
tf = ti() - t0

print("time to execute = ",tf)


# In[ ]:




