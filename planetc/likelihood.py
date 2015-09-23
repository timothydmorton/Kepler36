import numpy as np
import matplotlib.pyplot as plt
import pickle
import emcee
import triangle


from scipy.interpolate import LinearNDInterpolator as interpnd
from scipy.interpolate import interp2d, griddata
from scipy.optimize import fmin
from core_properties import get_core_iron, get_core_ice

mass = pickle.load(open('FinalMp.pkl'))
radius = pickle.load(open('FinalRp.pkl'))

initial_entropy = np.log10(pickle.load(open('InitialKH.pkl'))+0.01)
initial_mass = pickle.load(open('InitialMP.pkl'))
core_mass_grid = np.linspace(6.5, 10, 16)
core_radius_grid = np.linspace(1.4, 3.0, 24)

for i in range(0,16):
    for j in range(0,24):
        for k in range(0,64):
            for l in range(0,64):
                if initial_mass[i,j,k,l] > 0:
                    initial_mass[i,j,k,l]=np.log10((initial_mass[i,j,k,l]-core_mass_grid[i]*5.97219e27)/(core_mass_grid[i]*5.97219e27)+1e-10)


def mr_get(core_mass, core_radius, initial_mtot, initial_ent):
    
    if core_mass > core_mass_grid[-1] or \
        core_radius > core_radius_grid[-1] or \
        core_mass < core_mass_grid[0] or \
        core_radius < core_radius_grid[0]:
            return np.nan, np.nan
    
    mass_bin = np.digitize(np.atleast_1d(core_mass), core_mass_grid)[0]
    radius_bin = np.digitize(np.atleast_1d(core_radius), core_radius_grid)[0]
    
    mass_ilo = mass_bin - 1
    mass_ihi = mass_bin
    
    radius_ilo = radius_bin - 1
    radius_ihi = radius_bin

    #fns = []
    box_points = []
    box_mass_values = []
    box_radius_values = []
    for mi in [mass_ilo, mass_ihi]:
        for ri in [radius_ilo, radius_ihi]:            
            mass_slice = mass[mi, ri, :, :]
            radius_slice = radius[mi, ri, :, :]
            ent_grid = initial_entropy[mi, ri, :, :]
            minit_grid = initial_mass[mi, ri, :, :]
            points = np.array([ent_grid.ravel(),minit_grid.ravel()]).T

            mass_vals = mass_slice.ravel()
            mass_fn = interpnd(points, mass_vals, rescale=True)
            
            radius_vals = radius_slice.ravel()
            radius_fn = interpnd(points, radius_vals, rescale=True)

            box_points.append([core_mass_grid[mi],
                               core_radius_grid[ri]])
            box_mass_values.append([mass_fn(initial_ent, initial_mtot)])
            box_radius_values.append([radius_fn(initial_ent, initial_mtot)])
    
    box_points = np.array(box_points)
    box_mass_values = np.array(box_mass_values)
    box_radius_values = np.array(box_radius_values)
    
    return (griddata(box_points, box_mass_values, [core_mass, core_radius])[0][0],
            griddata(box_points, box_radius_values, [core_mass, core_radius])[0][0])
            

def loglike(p):
    
	mod_m, mod_r  = mr_get(*p)
	kep36_mass = 8.08 * 5.97219e27
	kep36_radius = 3.679 * 6.4e8
	kep36_mass_err = 0.03 * kep36_mass
	kep36_radius_err = 0.005 * kep36_radius

	lnlike = (-0.5*((kep36_mass - mod_m)**2/kep36_mass_err**2.0) +
            -0.5*((kep36_radius - mod_r)**2/kep36_radius_err**2.0))
    
	if np.isnan(lnlike):
		lnlike = -np.inf

    # include priors on core radius and core density

	Riron=get_core_iron(p[0],0.66666)
	Rice =get_core_ice (p[0],0.66666)

	if p[1] > Rice or \
		p[1] < Riron or \
		p[3] < 6.0 or \
		p[3] > 8.5: 

			lnlike = -np.inf

	return lnlike


def negloglike(p):
 	nlnlike=-loglike(p)

 	return nlnlike


pstart=fmin(negloglike, (7.73010527e+00,   1.97766086e+00,   np.log10(0.2), np.log10(1.01441188e+07)))
print pstart
print loglike(pstart)

ndim, nwalkers = 4, 600
p0 = pstart
new_p0 = np.array([p0*(1 + np.random.rand(ndim)*.1) for i in range(nwalkers)])
sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, threads=28)
new_pos, prob, state = sampler.run_mcmc(new_p0,100)
sampler.reset()
sampler.run_mcmc(new_pos, 3000)

figure=triangle.corner(sampler.flatchain)
figure.savefig("triangle.png")
