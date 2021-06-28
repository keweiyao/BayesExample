#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from scipy.interpolate import interp2d
from scipy.optimize import brentq # for root finding

cen = '30-40'
fname = 'CollisionGeometry/'+cen+'.h5' # 300 by 300 x-y grid
dx = dy = 0.1 # fm
dxdx = dx*dy
f = h5py.File(fname, 'r') 
BinaryCollisionDensity = f['event_0/Ncoll_density'].value # spatial density of jet production 
MediumDensity = f['event_0/matter_density'].value # spatial density of the medium

Nx, Ny = BinaryCollisionDensity.shape
L = Nx*dx
x = np.linspace(-L/2., L/2., Nx)
y = np.linspace(-L/2., L/2., Ny)

InterpolateDenstiy = interp2d(x, y, MediumDensity.T,
                              bounds_error=False, fill_value=0.)

plt.contour(x, y, BinaryCollisionDensity, cmap=plt.cm.Reds)
plt.contour(x, y, MediumDensity, cmap=plt.cm.Blues)

plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()


Nsamples = 1000

# Step 1:
# pick a random location according to the BinaryCollisionDensity
Pairs = np.array([[(-L/2.+i*dx,-L/2.+j*dy) for j in range(Ny)] 
                   for i in range(Nx)]).reshape(-1,2)
Weights = BinaryCollisionDensity.flatten()
Coll_x, Coll_y = np.array(random.choices(Pairs, Weights,k=Nsamples)).T

# Step 2:
# pick a random angle
Phi = np.random.uniform(-np.pi, np.pi, Nsamples)
print(Coll_x, Coll_y, Phi)
# Now the Density on the trajectory is defined by:
def Dl(l, x0, y0, phi, D0):
    D = InterpolateDenstiy(x0+l*np.cos(phi), y0+l*np.sin(phi) )
    return D - D0

# Step 3:
# For each jet, use a bisection method to sovle for the path length,
# where it reaches the boundary of the medium, e.g., Density0=0.1
lmin = 0
lmax = 20
L = []
D0 = 0.1
for ix, iy, iphi in zip(Coll_x, Coll_y, Phi):
    if InterpolateDenstiy(ix, iy) < D0:
        L.append(0)
    else:
        L.append(brentq(Dl, lmin, lmax, args=(ix,iy,iphi,D0)))

plt.hist(L)
plt.xlabel("$L$ [fm]")
plt.ylabel("$dN/dL$ [fm]")
plt.show()

