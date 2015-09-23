# calculates the core density

# based on Fortney et al. (2007)ab fiting formula - note Errata here!!

from numpy import log10

Mearth=5.9736e27
Rearth=6.371e8

def get_core_iron(Mass,Xiron):
	
	#constants
	a=0.0592
	b=0.0975
	c=0.2337
	d=0.4938
	e=0.3102
	f=0.7932

	Xrock=1.0-Xiron

	R=(a*Xrock+b)*(log10(Mass))**2.0+(c*Xrock+d)*(log10(Mass))+(e*Xrock+f)

	return R

def get_core_ice(Mass,Xice):
	
	#constants
	a=0.0912
	b=0.1603
	c=0.3330
	d=0.7387
	e=0.4639
	f=1.1193

	R=(a*Xice+b)*(log10(Mass))**2.0+(c*Xice+d)*(log10(Mass))+(e*Xice+f)

	return R
