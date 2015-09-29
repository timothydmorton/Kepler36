import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

FOO = "Hi" #for weird __init__ bug purposes

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate_box(np.ndarray[DTYPE_t] pt, np.ndarray[DTYPE_t, ndim=2] corners,
                       np.ndarray[DTYPE_t] values):
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

    cdef int N = len(pt)
    cdef DTYPE_t xd
    cdef DTYPE_t res
    if N==1:
        xd = (pt[0] - corners[0,0]) / (corners[0,1] - corners[0,0])
        res = values[0]*(1 - xd) + values[1]*xd
        return res

    cdef np.ndarray[DTYPE_t] new_values = np.empty(2**(N-1), dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=2] x_corners = np.array([[corners[0,0], 
                                                            corners[0,1]]])
    cdef int jump = 2**(N-1)
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t] pt0 = np.array([pt[0]])
    cdef np.ndarray[DTYPE_t] vals0 = np.empty(2, dtype=float)
    for i in xrange(2**(N-1)):
        vals0 = np.array([values[i], values[i+jump]])
        new_values[i] = interpolate_box(pt0, x_corners, vals0)
    
    cdef np.ndarray[DTYPE_t, ndim=2] new_corners = np.empty((N-1, 2), dtype=float)
    cdef np.ndarray[DTYPE_t] new_pt = np.empty(N-1, dtype=float)
    for i in xrange(N-1):
        new_corners[i,0] = corners[i+1, 0]
        new_corners[i,1] = corners[i+1, 1]
        new_pt[i] = pt[i+1]

    return interpolate_box(new_pt, new_corners, new_values)
        

@cython.boundscheck(False)
@cython.wraparound(False)
def interp_4d(np.ndarray[DTYPE_t] pt,
              np.ndarray[DTYPE_t] arr1, np.ndarray[DTYPE_t] arr2,
              np.ndarray[DTYPE_t] arr3, np.ndarray[DTYPE_t] arr4,
              np.ndarray[DTYPE_t, ndim=4] values):
    """
    pt: np.ndarray, length 4
    arr1, arr2, arr3, arr4: arrays that describe the different dimensions of values
    values: 4-d array (len(arr1) x len(arr2) x len(arr3) x len(arr4)) of values
    
    """


    #find the boundaries of the bounding box
    cdef np.ndarray[DTYPE_t, ndim=2] corners = np.empty((4,2), dtype=float)

    cdef int ilo1, ilo2, ilo3, ilo4
    cdef int ihi1, ihi2, ihi3, ihi4

    cdef int i1,i2,i3,i4

    cdef int n1 = len(arr1)
    cdef int n2 = len(arr2)
    cdef int n3 = len(arr3)
    cdef int n4 = len(arr4)

    cdef unsigned int i
    for i in xrange(n1):
        if arr1[i] > pt[0]:
            break
        i1 = i
    if i1 == n1 - 1:
        i1 -= 1
    corners[0,0] = arr1[i1]
    corners[0,1] = arr1[i1+1]

    for i in xrange(n2):
        if arr2[i] > pt[1]:
            break
        i2 = i
    if i2 == n2 - 1:
        i2 -= 1
    corners[1,0] = arr2[i2]
    corners[1,1] = arr2[i2+1]

    for i in xrange(n3):
        if arr3[i] > pt[2]:
            break
        i3 = i
    if i3 == n3 - 1:
        i3 -= 1
    corners[2,0] = arr3[i3]
    corners[2,1] = arr3[i3+1]

    for i in xrange(n4):
        if arr4[i] > pt[3]:
            break
        i4 = i
    if i4 == n4 - 1:
        i4 -= 1
    corners[3,0] = arr4[i4]
    corners[3,1] = arr4[i4+1]

              
    cdef np.ndarray[DTYPE_t] box_values = np.empty(16, dtype=float)
    
    cdef int j,k,l,m
    for j in xrange(2):
        for k in xrange(2):
            for l in xrange(2):
                for m in xrange(2):
                    box_values[j*8 + k*4 + l*2 + m] = values[i1+j,
                                                             i2+k,
                                                             i3+l,
                                                             i4+m]

    return interpolate_box(pt, corners, box_values)
