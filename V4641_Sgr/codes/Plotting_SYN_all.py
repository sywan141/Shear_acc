import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from naima.models import TableModel
from naima.models import InverseCompton
from NAIMA_RAD import Elect_spec, NAIMA_IC, NAIMA_SYN
from scipy.integrate import simps
from scipy.stats import chi2
from functools import partial
import warnings
import datetime
warnings.filterwarnings("ignore")

m_e = (const.m_e.cgs).value
c= (const.c.cgs).value
e = (const.e.esu).value
sigma_sb = (const.sigma_sb.cgs).value
sigma_e = (8*np.pi/3)*np.square(e**2/(m_e*c**2)) 
m_p = (const.m_p.cgs).value
len_jet=100.0
D_jet = ((len_jet*(u.pc)).cgs).value
pi = np.pi
inf = np.inf
nan = float('nan')
q=5/3
T_cmb = 2.72 # CMB temperature
z=0
xi=1
d_pc = ((6.2*(u.kpc)).to(u.pc)).value
start = datetime.datetime.now()
dir_dat1 = '/home/wsy/V4641_Sgr/read_flatsamples/'
dir_tar = '/home/wsy/V4641_Sgr/PDF_files/'
tau = 0.3
S_bxrism = 7*10**(-15)

#-------------------------------------------------------------------------------------------------------
# read observed data
datadir = '/home/wsy/V4641_Sgr/data/'

file1 = np.loadtxt(datadir+'HAWC.txt') 
file2 = np.loadtxt(datadir+'LHAASO.txt') 
file3 = np.loadtxt(datadir+'HESS.txt') 
# HAWC, in TeV
x1 = (file1[1:,0]*(u.TeV)).value
y1 = ((file1[1:,1]*(u.TeV)).to(u.erg)).value
upper1 = ((file1[1:,2]*(u.TeV)).to(u.erg)).value   # upper error
lower1 = ((-file1[1:,3]*(u.TeV)).to(u.erg)).value  # lower error
error_limit1 = [lower1,upper1]
err1 = (upper1+lower1)/2 # average
a1 = np.column_stack((x1,y1,err1))
#LHAASO,in TeV
x2 = (file2[2:,0]*(u.TeV)).value
y2 = ((file2[2:,1]*(u.TeV)).to(u.erg)).value
upper2 = ((file2[2:,2]*(u.TeV)).to(u.erg)).value
lower2 = ((file2[2:,3]*(u.TeV)).to(u.erg)).value
error_limit2 = [lower2,upper2]
err2 = (upper2+lower2)/2
a2 = np.column_stack((x2,y2,err2))
# HESS, in erg
x3 = (file3[:,0]*(u.TeV)).value
y3 = ((file3[:,1]*(u.erg))).value
upper3 = (file3[:,2]*(u.erg)).value-y3
lower3 = y3-(file3[:,3]*(u.erg)).value
error_limit3 = [lower3,upper3]
err3 = (upper3+lower3)/2
a3 = np.column_stack((x3,y3,err3))
a_all = np.row_stack((a1,a2,a3))
sorted_a_all = a_all[np.argsort(a_all[:, 0])]

ene_obs = np.float64(sorted_a_all[:,0])
flux_obs = np.float64(sorted_a_all[:,1])
err_obs = np.float64(sorted_a_all[:,2])

#-------------------------------------------------------------------------------------------------------
# plot syn radiation
#ene_syn = np.logspace(1,15,1000)
fig = plt.figure(figsize=(10, 8))
# 左上
file1 = np.loadtxt(dir_dat1+'SYN_5.0pc_spec.txt')
file1s = np.loadtxt(dir_dat1+'SYN_5.0pc_intflux.txt')
ene_syn = file1[:,0]
flux_syn = file1[:,1]
sed_min = file1[:,2]
sed_max = file1[:,3]
int_flux_min1 = np.min(file1s[:,0])
int_flux_max1 = np.max(file1s[:,0])
int_flux_min2 = np.min(file1s[:,1])
int_flux_max2 = np.max(file1s[:,1])
R=5.0
R_jet = ((R*(u.pc)).cgs).value
ax1 = fig.add_subplot(2, 2, 1) 
ax1.set_xlim(1e2,1e5)
ax1.set_ylim(1e-15,1e-11)
ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax1.set_title(r'Hard X-ray synchrotron ($R_{jet}$ = 5.0 pc)')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.plot(ene_syn,flux_syn,c='red',label='Best fit values')
ax1.fill_between(ene_syn, sed_min, sed_max, color="lightcoral", label=r" SEDs in 1$ \sigma$")
arcmin_2 = (2*R/(d_pc))*3437.7467707849396**2*(len_jet/d_pc)
Flux_max = S_bxrism*arcmin_2
ax1.hlines(Flux_max,2e3,1e4,colors='deepskyblue',linewidth=3,label=r'Intergrated flux from XRISM($S_B \Delta \Omega$)')
#ax1.hlines(4.4757530141016927e-14,2e3,1e4,colors='cyan',linestyle = '--',linewidth=2,label=r'Intergrated flux for the best-fit parameters (2 - 10 keV)')

#右上
file2 = np.loadtxt(dir_dat1+'SYN_1.0pc_spec.txt')
file2s = np.loadtxt(dir_dat1+'SYN_1.0pc_intflux.txt')
ene_syn = file2[:,0]
flux_syn = file2[:,1]
sed_min = file2[:,2]
sed_max = file2[:,3]
int_flux_min1 = np.min(file2s[:,0])
int_flux_max1 = np.max(file2s[:,0])
int_flux_min2 = np.min(file2s[:,1])
int_flux_max2 = np.max(file2s[:,1])
R=1.0
R_jet = ((R*(u.pc)).cgs).value
ax2 = fig.add_subplot(2, 2, 2) 
ax2.set_xlim(1e2,5e5)
ax2.set_ylim(1e-14,1e-11)
ax2.set_xlabel('Energy (eV)')
ax2.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax2.set_title(r'Hard X-ray synchrotron ($R_{jet}$ = 1.0 pc)')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.plot(ene_syn,flux_syn,c='red',label='Best fit values')
ax2.fill_between(ene_syn, sed_min, sed_max, color="lightcoral", label=r" SEDs in 1$ \sigma$")
arcmin_2 = (2*R/(d_pc))*3437.7467707849396**2*(len_jet/d_pc)
Flux_max = S_bxrism*arcmin_2
ax2.hlines(Flux_max,2e3,1e4,colors='deepskyblue',linewidth=3,label=r'Intergrated flux from XRISM($S_B \Delta \Omega$)')
#ax2.hlines(1.5434403927018393e-12,2e3,1e4,colors='cyan',linestyle = '--',linewidth=2,label=r'Intergrated flux for the best-fit parameters (2 - 10 keV)')

#左下
file3 = np.loadtxt(dir_dat1+'SYN_0.5pc_spec.txt')
file3s = np.loadtxt(dir_dat1+'SYN_0.5pc_intflux.txt')
ene_syn = file3[:,0]
flux_syn = file3[:,1]
sed_min = file3[:,2]
sed_max = file3[:,3]
int_flux_min1 = np.min(file3s[:,0])
int_flux_max1 = np.max(file3s[:,0])
int_flux_min2 = np.min(file3s[:,1])
int_flux_max2 = np.max(file3s[:,1])
R=0.5
R_jet = ((R*(u.pc)).cgs).value
ax3 = fig.add_subplot(2, 2, 3) 
ax3.set_xlim(1e2,1e6)
ax3.set_ylim(1e-15,1e-10)
ax3.set_xlabel('Energy (eV)')
ax3.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax3.set_title(r'Hard X-ray synchrotron ($R_{jet}$ = 0.5 pc)')
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.plot(ene_syn,flux_syn,c='red',label='Best fit values')
ax3.fill_between(ene_syn, sed_min, sed_max, color="lightcoral", label=r" SEDs in 1$ \sigma$")
arcmin_2 = (2*R/(d_pc))*3437.7467707849396**2*(len_jet/d_pc)
Flux_max = S_bxrism*arcmin_2
ax3.hlines(Flux_max,2e3,1e4,colors='deepskyblue',linewidth=3,label=r'Intergrated flux from XRISM($S_B \Delta \Omega$)')
#ax3.hlines(5.481933534531946e-12,2e3,1e4,colors='cyan',linestyle = '--',linewidth=2,label=r'Intergrated flux for the best-fit parameters (2 - 10 keV)')

file4 = np.loadtxt(dir_dat1+'SYN_0.1pc_spec.txt')
file4s = np.loadtxt(dir_dat1+'SYN_0.1pc_intflux.txt')
ene_syn = file4[:,0]
flux_syn = file4[:,1]
sed_min = file4[:,2]
sed_max = file4[:,3]
int_flux_min1 = np.min(file4s[:,0])
int_flux_max1 = np.max(file4s[:,0])
int_flux_min2 = np.min(file4s[:,1])
int_flux_max2 = np.max(file4s[:,1])
R=0.1
R_jet = ((R*(u.pc)).cgs).value
ax4 = fig.add_subplot(2, 2, 4) 
ax4.set_xlim(1e2,1e7)
ax4.set_ylim(1e-15,8e-10)
ax4.set_xlabel('Energy (eV)')
ax4.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax4.set_title(r'Hard X-ray synchrotron ($R_{jet}$ = 0.1 pc)')
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.plot(ene_syn,flux_syn,c='red',label='Best fit values')
ax4.fill_between(ene_syn, sed_min, sed_max, color="lightcoral", label=r" SEDs in 1$ \sigma$")
arcmin_2 = (2*R/(d_pc))*3437.7467707849396**2*(len_jet/d_pc)
Flux_max = S_bxrism*arcmin_2
ax4.hlines(Flux_max,2e3,1e4,colors='deepskyblue',linewidth=3,label=r'Intergrated flux from XRISM($S_B \Delta \Omega$)')
#ax4.hlines(8.123932466823229e-11,2e3,1e4,colors='cyan',linestyle = '--',linewidth=2, label=r'Intergrated flux for the model')
plt.tight_layout()
plt.legend(ncol=1, frameon=False,fontsize = 8)
plt.show()
plt.savefig(dir_tar+ 'SYN_all_HESS_flat.pdf',format="pdf", dpi=300, bbox_inches="tight")
plt.savefig(dir_tar+ 'SYN_all_HESS_flat.png')