import numpy as np
import astropy.io.fits as fits
import astro_constants_cgs as const

dlist=fits.open('1223269s4.fits')

R3=1.652*const.rsun
T3=4525.3
ab=0.1151*const.au
ac=0.1280*const.au

Tb=T3*np.sqrt(R3/(2.*ab))
Tc=T3*np.sqrt(R3/(2.*ac))

csb=np.sqrt(const.kb*Tb/(2.35*const.amu))
csc=np.sqrt(const.kb*Tc/(2.35*const.amu))

data=dlist[1].data
samples=np.array([data.field(0)/(60.*60.*24.)**2.,data.field(1),data.field(2),data.field(3),data.field(4),data.field(5),
                  data.field(6),data.field(7),data.field(8),data.field(9),data.field(10),data.field(11),data.field(12),
                  data.field(13),data.field(14)*const.au,data.field(15),data.field(16),data.field(17),data.field(18),
                  data.field(19)])
samples=np.transpose(samples)

#convert Qp --> Mb
#convert Qm --> Mc
#convert Rb/Rs --> Rb
#convert Rc/Rs --> Rc

Mstar=samples[:,0]/const.g*(4./3.*np.pi*samples[:,14]**3.0)
Mb=samples[:,1]*Mstar/(1.+1./samples[:,2])
Mc=Mb/samples[:,2]
Rb=samples[:,15]*samples[:,14]
Rc=samples[:,16]*samples[:,14]
samples[:,1]=Mb
samples[:,2]=Mc
samples[:,15]=Rb
samples[:,16]=Rc

Mb=Mb/const.mearth
Mc=Mc/const.mearth
Rb=Rb/const.rearth
Rc=Rc/const.rearth

samples_use=np.array([Mb,Mc,Rb,Rc])

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
freq=gaussian_kde(samples_use)

import pickle

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

from scipy.interpolate import LinearNDInterpolator as interpnd
from scipy.interpolate import interp2d, griddata
from core_properties import get_core_iron, get_core_ice

def mr_get_b(core_mass, core_comp, initial_mtot, initial_ent):
    
    if core_comp < -1. or core_comp > 1.:
        return np.nan, np.nan, np.nan
    if core_comp < 0.:
        core_radius=get_core_iron(core_mass,np.abs(core_comp))
    else:
        core_radius=get_core_ice(core_mass,np.abs(core_comp))
    if core_mass > core_mass_grid_b[-1] or \
        core_radius > core_radius_grid_b[-1] or \
        core_mass < core_mass_grid_b[0] or \
        core_radius < core_radius_grid_b[0]:
            return np.nan, np.nan, np.nan
  
	# replace when new grid finishes running  
    if initial_mtot < np.log10(2.001e-2):
        return core_mass*const.mearth, core_radius*const.rearth, core_radius*const.rearth

    mass_bin = np.digitize(np.atleast_1d(core_mass), core_mass_grid_b)[0]
    radius_bin = np.digitize(np.atleast_1d(core_radius), core_radius_grid_b)[0]
    
    mass_ilo = mass_bin - 1
    mass_ihi = mass_bin
    
    radius_ilo = radius_bin - 1
    radius_ihi = radius_bin

    #fns = []
    box_points = []
    box_mass_values = []
    box_radius_values = []
    box_iradius_values = []
    for mi in [mass_ilo, mass_ihi]:
        for ri in [radius_ilo, radius_ihi]:            
            mass_slice = mass_b[mi, ri, :, :]
            radius_slice = radius_b[mi, ri, :, :]
            ir_slice = initial_radius_b[mi, ri, :, :]
            ent_grid = initial_entropy_b[mi, ri, :, :]
            minit_grid = initial_mass_b[mi, ri, :, :]
            points = np.array([ent_grid.ravel(),minit_grid.ravel()]).T

            mass_vals = mass_slice.ravel()
            
            mass_fn = interpnd(points, mass_vals, rescale=True)
            
            radius_vals = radius_slice.ravel()
            radius_fn = interpnd(points, radius_vals, rescale=True)

            iradius_vals = ir_slice.ravel()
            iradius_fn = interpnd(points,iradius_vals, rescale=True)
            
            box_points.append([core_mass_grid_b[mi],
                               core_radius_grid_b[ri]])
            box_mass_values.append([mass_fn(initial_ent, initial_mtot)])
            box_radius_values.append([radius_fn(initial_ent, initial_mtot)])
            box_iradius_values.append([iradius_fn(initial_ent, initial_mtot)])
    
    box_points = np.array(box_points)
    box_mass_values = np.array(box_mass_values)
    box_radius_values = np.array(box_radius_values)
    box_iradius_values = np.array(box_iradius_values)
    
    return (griddata(box_points, box_mass_values, [core_mass, core_radius])[0][0],
            griddata(box_points, box_radius_values, [core_mass, core_radius])[0][0],
            griddata(box_points,box_iradius_values, [core_mass, core_radius])[0][0])

def mr_get_c(core_mass, core_comp, initial_mtot, initial_ent):
    
    if core_comp < -1. or core_comp > 1.:
        return np.nan, np.nan, np.nan
    if core_comp < 0.:
        core_radius=get_core_iron(core_mass,np.abs(core_comp))
    else:
        core_radius=get_core_ice(core_mass,np.abs(core_comp))
    
    if core_mass > core_mass_grid_c[-1] or \
        core_radius > core_radius_grid_c[-1] or \
        core_mass < core_mass_grid_c[0] or \
        core_radius < core_radius_grid_c[0]:
            return np.nan, np.nan, np.nan
    
    mass_bin = np.digitize(np.atleast_1d(core_mass), core_mass_grid_c)[0]
    radius_bin = np.digitize(np.atleast_1d(core_radius), core_radius_grid_c)[0]
    
    mass_ilo = mass_bin - 1
    mass_ihi = mass_bin
    
    radius_ilo = radius_bin - 1
    radius_ihi = radius_bin

    #fns = []
    box_points = []
    box_mass_values = []
    box_radius_values = []
    box_iradius_values = []
    for mi in [mass_ilo, mass_ihi]:
        for ri in [radius_ilo, radius_ihi]:            
            mass_slice = mass_c[mi, ri, :, :]
            radius_slice = radius_c[mi, ri, :, :]
            ir_slice = initial_radius_c[mi, ri, :, :]
            ent_grid = initial_entropy_c[mi, ri, :, :]
            minit_grid = initial_mass_c[mi, ri, :, :]
            points = np.array([ent_grid.ravel(),minit_grid.ravel()]).T

            mass_vals = mass_slice.ravel()
            mass_fn = interpnd(points, mass_vals, rescale=True)
            
            radius_vals = radius_slice.ravel()
            radius_fn = interpnd(points, radius_vals, rescale=True)

            iradius_vals = ir_slice.ravel()
            iradius_fn = interpnd(points,iradius_vals, rescale=True)
            
            box_points.append([core_mass_grid_c[mi],
                               core_radius_grid_c[ri]])
            box_mass_values.append([mass_fn(initial_ent, initial_mtot)])
            box_radius_values.append([radius_fn(initial_ent, initial_mtot)])
            box_iradius_values.append([iradius_fn(initial_ent, initial_mtot)])
    
    box_points = np.array(box_points)
    box_mass_values = np.array(box_mass_values)
    box_radius_values = np.array(box_radius_values)
    box_iradius_values = np.array(box_iradius_values)
    
    return (griddata(box_points, box_mass_values, [core_mass, core_radius])[0][0],
            griddata(box_points, box_radius_values, [core_mass, core_radius])[0][0],
            griddata(box_points,box_iradius_values, [core_mass, core_radius])[0][0])

def loglike(p):
    # get the final mass and radius from the model
    if p[5] < 6.0 or \
    p[5] > 9.75: 
        return -np.inf

    if p[2] < -3:
    	return -np.inf
    
    pb=(p[0],p[4],p[2],p[5])
    pc=(p[1],p[4],p[3],p[5])
    
    mod_mc, mod_rc, i_rc  = mr_get_c(*pc)
    mod_mb, mod_rb, i_rb  = mr_get_b(*pb)
    
    
    if np.isnan(mod_mb):
        return -np.inf
    if (np.isnan(mod_mc)):
        return -np.inf
    
    lnlike=np.log(freq([mod_mb/const.mearth,mod_mc/const.mearth,
                 mod_rb/const.rearth,mod_rc/const.rearth])*const.mearth/mod_mc*(1.-mod_mb/mod_mc))
    
    #check initial radius not > 0.1 Rb as not bound above
    Rbondi_b=const.g*(1.+10**p[2])*p[0]*const.mearth/(2.*csb**2.)
    if i_rb > 0.1*Rbondi_b:
        lnlike=-np.inf
    Rbondi_b=const.g*(1.+10**p[3])*p[1]*const.mearth/(2.*csc**2.)
    if i_rc > 0.1*Rbondi_b:
        lnlike=-np.inf

    if np.isnan(lnlike):
        lnlike = -np.inf
    return lnlike
        
def negloglike(p):
    nlnlike=-loglike(p)
    return nlnlike


if __name__=='__main__':

    #from scipy.optimize import fmin
    #pstart=fmin(negloglike, (4.45, 7.75, np.log10(0.02), np.log10(0.1),  -0.333, np.log10(1.01441188e+08)))
    #print pstart
    #print loglike(pstart)

    pstart=(4.64, 7.61, -1.688, -0.69, -0.3, 8.0)

    print "starting MCMC"

    import emcee
    ndim, nwalkers = 6, 100
    p0 = pstart
    new_p0 = np.array([p0*(1 + np.random.rand(ndim)*.05) for i in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, threads=28)
    new_pos, prob, state = sampler.run_mcmc(new_p0,100)
    sampler.reset()
    sampler.run_mcmc(new_pos, 1000)

    file1=open('chain_long.pkl','wb')
    pickle.dump(sampler.flatchain,file1)
    file1.close()

    file2=open('chain_long2.pkl','wb')
    pickle.dump(sampler,file2)
    file2.close()
