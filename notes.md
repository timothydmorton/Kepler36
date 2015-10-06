To implement faster interpolation:

For given Mc, Rc, M_ini, E_ini:

* Find the bracketing grid values from Mc_grid and Rc_grid
* Find the ~10 closest M_ini, E_ini points by brute-force searching the slices
* Build 4-d interpolation object using this small number of points.
