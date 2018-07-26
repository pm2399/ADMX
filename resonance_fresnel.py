import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
import sys

c = constants.c
pi = constants.pi

def f00n(n, L, ro):
    resonant_frequency = (n+1)*c/(2*L) + c/(4*L*pi)*np.arccos(1-2*L/ro)
    return resonant_frequency

def fresnel_n(a, L, f):
    fresnel_number = a**2*f/(c*L)
    return fresnel_number
   
ro = 0.33
n = 19
a = 0.15

L = np.linspace(0.169, 0.2026)
f = f00n(n, L, ro)
N = fresnel_n(a, L, f)

fig, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(L*1e2, f/1e9)
axarr[0].grid()
axarr[1].plot(L*1e2, N)
axarr[1].grid()
axarr[1].set_xlabel('Cavity Length [cm]')
axarr[0].set_ylabel(r'$f_{00-19}$ [GHz]')
axarr[1].set_ylabel('Fresnel Number')
plt.tight_layout()
plt.savefig('TEM_00_19.pdf', bbox_inches='tight')
plt.close()
