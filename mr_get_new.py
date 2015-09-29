import numpy as np
from numba import jit
import pickle

"""
mass_b = pickle.load(open('planetb/FinalMp.pkl'))
radius_b = pickle.load(open('planetb/FinalRp.pkl'))
initial_entropy_b = np.log10(pickle.load(open('planetb/InitialKH.pkl'))+0.01)
initial_mass_b = pickle.load(open('planetb/InitialMP.pkl'))
initial_radius_b = pickle.load(open('planetb/InitialRp.pkl'))

mass_c = pickle.load(open('planetc/FinalMp.pkl'))
radius_c = pickle.load(open('planetc/FinalRp.pkl'))
initial_entropy_c = np.log10(pickle.load(open('planetc/InitialKH.pkl'))+0.01)
initial_mass_c = pickle.load(open('planetc/InitialMP.pkl'))
initial_radius_c = pickle.load(open('planetc/InitialRp.pkl'))

core_mass_grid_b = np.linspace(3.8, 5.2, 16)
core_radius_grid_b = np.linspace(1.39, 1.58, 16)

core_mass_grid_c = np.linspace(6.5, 10, 16)
core_radius_grid_c = np.linspace(1.4, 3.0, 24)

for i in range(0,16):
    for j in range(0,24):
        for k in range(0,64):
            for l in range(0,64):
                if initial_mass_c[i,j,k,l] > 0:
                    initial_mass_c[i,j,k,l]=np.log10((initial_mass_c[i,j,k,l]-core_mass_grid_c[i]*5.97219e27)/(core_mass_grid_c[i]*5.97219e27)+1e-10)
                    
for i in range(0,16):
    initial_mass_b[i,:,:,:]=np.log10((initial_mass_b[i,:,:,:]-core_mass_grid_b[i]*5.97219e27)/(core_mass_grid_b[i]*5.97219e27)+1e-10)
initial_mass_b[np.isnan(initial_mass_b)]=0.0
"""

a1 = np.arange(10)*2.
a2 = np.arange(11)*3.
a3 = np.arange(12)*4.
a4 = np.arange(13)*5.

v = np.zeros((10,11,12,13))

for i in range(len(a1)):
    for j in range(len(a2)):
        for k in range(len(a3)):
            for l in range(len(a4)):
                v[i,j,k,l] = a1[i]*a2[j]*a3[k]*a4[l]


@jit(nopython=True)
def linear_interpolate(pt, corners, values):
    """Does linear interpolation given N-dimensional box surrounding pt

    Dimension (=N) is defined by length of point

    corners is (N x 2) array with the the coordinates of the box corners
    in form [[x0, x1], [y0, y1], [z0, z1], ...]

    values is a 1-D array of length 2*N, with indices interpreted
    in binary: e.g. for N=2, values = [v00=v(x0,y0), v01=v(x0,x1), 
                                       v10=v(x1,y1), v11=v(x1,y1)]

    
    This is implemented recursively; each iteration interpolating out the 
    first dimension.

    """
    N = len(pt)

    if N==1:
        xd = (pt[0] - corners[0]) / (corners[1] - corners[0])
        return values[0]*(1 - xd) + values[1]*xd

    #interpolate out the first dimension
    new_values = np.empty(2**(N-1), dtype=np.float64)
    x_corners = np.array([corners[0,0], corners[0,1]])
    jump = 2**(N-1)
    for i in range(2**(N-1)):
        new_values[i] = linear_interpolate(np.array([pt[0]]), x_corners, 
                                           np.array([values[i], values[i+jump]]))
    
    new_corners = np.empty((N-1, 2), dtype=np.float64)
    new_pt = np.empty(N-1, dtype=np.float64)
    for i in range(N-1):
        new_corners[i,0] = corners[i+1, 0]
        new_corners[i,1] = corners[i+1, 1]
        new_pt = pt[i+1]

    return linear_interpolate(new_pt, new_corners, new_values)
    
        

@jit(nopython=True)
def find_box_4d(pt, arr1, arr2, arr3, arr4, values):
    """
    pt is length-4 array, arr1-4 do not have to be the same size

    values is a 4-d array with axes defined by arr1-4

    returns two 4x2 arrays: box corner indices, values at corners

    """

    ilo_1 = 0; ihi_1 = 0
    ilo_2 = 0; ihi_2 = 0
    ilo_3 = 0; ihi_3 = 0
    ilo_4 = 0; ihi_4 = 0

    i_box = np.empty((4,4), dtype=np.float64)
    val_box = np.empty((4,4), dtype=np.float64)

    ind = 0
    for i1 in range(len(arr1)):
        if arr1[i1] > pt[0]:
            break
        ind = i1
    if ind == len(arr1) - 1:
        ind -= 1
    i_box[0,0] = ind
    i_box[0,1] = ind + 1
    val_box[0,0] = arr1[ind]
    val_box[0,1] = arr1[ind + 1]

    ind = 0
    for i2 in range(len(arr2)):
        if arr2[i2] > pt[1]:
            break
        ind = i2
    if ind == len(arr2) - 1:
        ind -= 1
    i_box[1,0] = ind
    i_box[1,1] = ind + 1
    val_box[1,0] = arr2[ind]
    val_box[1,1] = arr2[ind + 1]

    ind = 0
    for i3 in range(len(arr3)):
        if arr3[i3] > pt[2]:
            break
        ind = i3
    if ind == len(arr3) - 1:
        ind -= 1
    i_box[2,0] = ind
    i_box[2,1] = ind + 1
    val_box[2,0] = arr3[ind]
    val_box[2,1] = arr3[ind + 1]

    ind = 0
    for i4 in range(len(arr4)):
        if arr4[i4] > pt[3]:
            break
        ind = i4
    if ind == len(arr4) - 1:
        ind -= 1
    i_box[3,0] = ind
    i_box[3,1] = ind + 1
    val_box[3,0] = arr4[ind]
    val_box[3,1] = arr4[ind + 1]

    return i_box, val_box


    
    
