
import numpy as np
import matplotlib.pyplot as plt
from time import time as ti
from matplotlib import cm
from tqdm import tqdm
from numba import cuda
# import nvtx
import cupy as cp
import numba as nb

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

nvtxRangePush("macroscopic function")
@cuda.jit
def macroscopic(rho, u, fin, nx, ny, v):
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        rho[i, j] = 0.0
        u[i, j, 0] = 0.0
        u[i, j, 1] = 0.0
        for k in range(9):
            rho[i, j] += fin[k, i, j]
            u[i, j, 0] += fin[k, i, j] * v[k, 0]
            u[i, j, 1] += fin[k, i, j] * v[k, 1]
        u[i, j, 0] /= rho[i, j]
        u[i, j, 1] /= rho[i, j]
nvtxRangePop()

nvtxRangePush("equilibrium function")
@cuda.jit
def equilibrium(feq, rho, u, v, t):
    nx, ny, _ = u.shape
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        usqr = (3/2)*(u[i, j, 0]**2 + u[i, j, 1]**2)
        for k in range(9):
            cu = 3*(v[k, 0]*u[i, j, 0] + v[k, 1]*u[i, j, 1])
            feq[k, i, j] = rho*t[k]*(1 + cu + 0.5*cu**2 - usqr)
nvtxRangePop()

nvtxRangePush("obstacle function")
def obstacle_fun(cx, cy, r):
    def inner(x, y):
        return (x-cx)**2+(y-cy)**2<r**2
    return inner
nvtxRangePop()

nvtxRangePush("inivel function")
def inivel(uLB, ly, d, nx, ny):
    yy = cp.linspace(0, ny - 1, ny).repeat(nx).reshape((ny, nx))
    yy = yy.T
    vel = cp.zeros((d, nx, ny))
    for dir in range(d):
        vel[dir,:,:] = (1-dir) * uLB * (1 + 1e-4 * cp.sin(yy / ly * 2 * cp.pi))
    return vel
nvtxRangePop()

nvtxRangePush("compute_rho function")
def compute_rho(fin_col_1, fin_col_2, u_0):
    return 1 / (1 - u_0) * (cp.sum(fin_col_1, axis=0) + 2 * cp.sum(fin_col_2, axis=0))
nvtxRangePop()

nvtxRangePush("time_stepping function")
def time_stepping(feq, maxIter, nx, ny, obstacle, vel, v, t, omega):
    t0 = ti()
    for time in tqdm(range(maxIter)):
        fin = cp.zeros((9, nx, ny))
        # Outflow boundary condition (right side)
        fin[2, -1, :] = fin[2, -2, :]

        # Compute macroscopic variables
        rho = cp.zeros((nx, ny))
        u = cp.zeros((nx, ny, 2))
        macroscopic[blockspergrid, threadsperblock](rho, u, fin, nx, ny, v)
        
        # Inlet boundary condition (left wall)
        u[0, :, :] = vel[:, 0, :].T

        # Collide
        fout = fin - omega * (fin - feq)

        # Bounce-back
        for i in range(9):
            fout[i, obstacle] = fin[8 - i, obstacle]

        # Stream
        for i in range(9):
            fin[i, :, :] = cp.roll(cp.roll(fout[i, :, :], v[i, 0], axis=0), v[i, 1], axis=1)

        # Output an image every 100 iterations
        if time % 100 == 0:
            plt.clf()
            u_temp = u.get()
            plt.imshow(np.sqrt(u_temp[0]**2 + u_temp[1]**2).T, cmap='Reds')
            plt.savefig(f"./numbaTest/vel{time//100:03d}.png")

    tf = ti() - t0
    print("Time to execute:", tf)
nvtxRangePop()

Re = 170.0
maxIter = 50000
nx, ny = 680, 240
ly = ny-1
uLB = 0.04
cx, cy, r = nx//4, ny//2, ny/9
nulb = uLB*r/Re
omega = 1 / (3*nulb+0.5)

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

col_0 = cp.array([0,1,2])
col_1 = cp.array([3,4,5])
col_2 = cp.array([6,7,8])

obstacle = cp.array(cp.fromfunction(obstacle_fun(cx,cy,r),(nx, ny)))
vel = inivel(uLB, ly, 2, nx, ny)
feq = cp.zeros((9, nx, ny))

threadsperblock = (16, 16)
blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

equilibrium[blockspergrid, threadsperblock](feq, 1, vel, v, t)

time_stepping(feq, maxIter, nx, ny, obstacle, vel, v, t, omega)
