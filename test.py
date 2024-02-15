import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from time import time as ti
from matplotlib import cm
from tqdm import tqdm
from numba import jit, cuda

# Replace NumPy arrays with CuPy arrays:
fin_gpu = cp.asarray(fin)
rho_gpu = cp.asarray(rho)
u_gpu = cp.asarray(u)
fout_gpu = cp.asarray(fout)

@cuda.jit(device=True)
def macroscopic(fin, nx, ny, v):
    rho = cp.sum(fin,axis=0)
    u = cp.zeros((2,nx,ny))
    for i in range(9):
        u[0,:,:] += v[i,0]*fin[i,:,:]
        u[1,:,:] += v[i,1]*fin[i,:,:]
    u /= rho
    return rho, u

@cuda.jit(device=True)
def equilibrium(rho, u, v, t, nx, ny):
    usqr = (3/2)*(u[0]**2+u[1]**2)
    feq = cp.zeros((9,nx,ny))
    for i in range(9):
        cu = 3*(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*t[i]*(1+cu+0.5*cu**2-usqr)
    return feq

@cuda.jit(device=True)
def outflow_boundary(fin):
    col_2 = fin.shape[0] - 1
    col_size = fin.shape[1]
    for i in range(col_size):
        fin[col_2, i, :] = fin[col_2, i - 1, :]

@cuda.jit(device=True)
def inlet_boundary(u, rho, vel, fin):
    col_1 = 0
    col_2 = fin.shape[0] - 1
    row_size = fin.shape[2]
    for i in range(row_size):
        u[0, i, :] = vel[0, i, :]
        rho[0, i] = 1 / (1 - u[0, 0, i]) * (cp.sum(fin[col_1, 0, i], axis=0) +
                                            2 * cp.sum(fin[col_2, 0, i], axis=0))

@cuda.jit(device=True)
def collide_and_bounceback(fin, feq, fout, omega, obstacle):
    for i in range(9):
        fout[i, obstacle] = fin[8 - i, obstacle]
        fout[i, :, :] = fin[i, :, :] - omega * (fin[i, :, :] - feq[i, :, :])

@cuda.jit(device=True)
def stream(fin, fout, v):
    for i in range(9):
        fin[i, :, :] = cuda.atomic.roll(cuda.atomic.roll(fout[i, :, :], v[i, 0], axis=0), v[i, 1], axis=1)

def obstacle_fun(cx, cy, r):
    def inner(x, y):
        return (x-cx)**2+(y-cy)**2<r**2
    return inner

def inivel( uLB, ly):
    def inner(d,x,y):
        return (1-d) * uLB * (1+1e-4*np.sin(y/ly*2*np.pi))
    return inner

# Constants
Re = 10.0
maxIter = 70000
nx, ny = 420, 180
ly = ny - 1
uLB = 0.04
cx, cy, r = nx // 4, ny // 2, ny / 9
nulb = uLB * r / Re
omega = 1 / (3 * nulb + 0.5)
v = cp.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]])
t = cp.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
col_0 = cp.array([0, 1, 2])
col_1 = cp.array([3, 4, 5])
col_2 = cp.array([6, 7, 8])
obstacle = cp.fromfunction(obstacle_fun(cx, cy, r), (nx, ny))
vel = cp.fromfunction(inivel(uLB, ly), (2, nx, ny))
fin = equilibrium(1, vel, v, t, nx, ny)

# Main simulation loop
@cuda.jit
def simulate(maxIter, fin, nx, ny, v, vel, omega, obstacle):
    blockDim = (8, 8)  # Choose an appropriate block size
    gridDim = ((ny + blockDim[0] - 1) // blockDim[0], (nx + blockDim[1] - 1) // blockDim[1])
    
    for time in range(maxIter):
        outflow_boundary[gridDim, blockDim](fin)
        rho, u = macroscopic(fin, nx, ny, v)
        inlet_boundary[gridDim, blockDim](u, rho, vel, fin)
        
        # Compute equilibrium on CPU
        feq = equilibrium(rho, u, v, t, nx, ny)
        
        # Transfer feq to GPU
        d_feq = cuda.to_device(feq)
        
        # Call collide_and_bounceback kernel
        collide_and_bounceback[gridDim, blockDim](fin, d_feq, fout, omega, obstacle)
        
        stream[gridDim, blockDim](fin, fout, v)
        if time % 100 == 0:
            plt.clf()
            plt.imshow(cp.asnumpy(cp.sqrt(u[0]**2 + u[1]**2)).T, cmap=cm.coolwarm)
            plt.savefig("vel{0:03d}.png".format(time//100))

t0 = ti()
simulate[maxIter, (1,), (1,)](maxIter, fin, nx, ny, v, vel, omega, obstacle)
tf = ti() - t0
print("Time to execute =", tf)



# Parallelize element-wise operations in loops:
for time in tqdm(range(maxIter)):
    # ... (other code)

    # Parallelize macroscopic calculation using vectorized operations:
    rho_gpu[:] = cp.sum(fin_gpu, axis=0)
    u_gpu[0, :] = cp.dot(v[:, 0], fin_gpu[:, 0, :]) * 3 / rho_gpu[:]
    u_gpu[1, :] = cp.dot(v[:, 1], fin_gpu[:, 0, :]) * 3 / rho_gpu[:]
    u_gpu[:] /= rho_gpu

    # Parallelize equilibrium calculation using vectorized operations:
    usqr_gpu = 3 / 2 * (u_gpu[0, :]**2 + u_gpu[1, :]**2)
    feq_gpu = rho_gpu * t * (1 + cp.einsum("i,ij->ij", cu, u_gpu) + 0.5 * cu**2 - usqr_gpu)

