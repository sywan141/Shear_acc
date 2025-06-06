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

R = 5
m_e = (const.m_e.cgs).value
c= (const.c.cgs).value
e = (const.e.esu).value
sigma_sb = (const.sigma_sb.cgs).value
sigma_e = (8*np.pi/3)*np.square(e**2/(m_e*c**2)) 
m_p = (const.m_p.cgs).value
R_jet = ((R*(u.pc)).cgs).value
D_jet = ((100*(u.pc)).cgs).value
pi = np.pi
inf = np.inf
nan = float('nan')
q=5/3
T_cmb = 2.72 # CMB temperature
z=0
xi=1
d_pc = ((6.2*(u.kpc)).to(u.pc)).value
start = datetime.datetime.now()
dir_dat1 = '/home/wsy/V4641_Sgr/IC_noHESS/'
dir_tar = '/home/wsy/V4641_Sgr/PDF_files/'

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
a_all = np.row_stack((a1,a2))
sorted_a_all = a_all[np.argsort(a_all[:, 0])]

ene_obs = np.float64(sorted_a_all[:,0])
flux_obs = np.float64(sorted_a_all[:,1])
err_obs = np.float64(sorted_a_all[:,2])

#-------------------------------------------------------------------------------------------------------
# noHESS fits, 5pc
fig = plt.figure(figsize=(10, 8))
# 左上
file1 = np.loadtxt(dir_dat1+'IC_5.0pc_spec.txt')
spectrum_energy_ic = file1[:,0]
IC_tot = file1[:,1]
IC_cmb = file1[:,2]
IC_fir = file1[:,3]
IC_nir = file1[:,4]
R_jet = ((5.0*(u.pc)).cgs).value
ax1 = fig.add_subplot(2, 2, 1) 
ax1.set_xlim(1e1,5e3)
ax1.set_ylim(3e-15,1e-10)
ax1.set_xlabel('Energy (TeV)')
ax1.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax1.set_title(r'Inverse Compton ($R_{jet}$ = 5.0 pc)')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.plot(spectrum_energy_ic,IC_tot,c='blue',label='Best fit values(Total)')
ax1.plot(spectrum_energy_ic,IC_cmb,c='C1',linestyle='--', label='Best fit values(CMB)')
ax1.plot(spectrum_energy_ic,IC_fir,c='C2',linestyle='--', label='Best fit values(FIR)')
#ax1.plot(spectrum_energy_ic,IC_nir,c='cyan',linestyle='--',label='Best fit values(NIR)')
ax1.errorbar(x1,y1,c='darkviolet',fmt = 'o', yerr = error_limit1, ls='None',label = 'HAWC')
ax1.errorbar(x2,y2,c='firebrick',fmt = 'D', yerr = error_limit2, ls='None',label = 'LHAASO')
ax1.text(x=0.05,y=0.05,s=r'$\chi^2/d.o.f$=4.94/6',transform=ax1.transAxes)
#右上
file2 = np.loadtxt(dir_dat1+'IC_1.0pc_spec.txt')
spectrum_energy_ic = file2[:,0]
IC_tot = file2[:,1]
IC_cmb = file2[:,2]
IC_fir = file2[:,3]
IC_nir = file2[:,4]
R_jet = ((1.0*(u.pc)).cgs).value
ax2 = fig.add_subplot(2, 2, 2) 
ax2.set_xlim(1e1,5e3)
ax2.set_ylim(3e-15,1e-10)
ax2.set_xlabel('Energy (TeV)')
ax2.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax2.set_title(r'Inverse Compton ($R_{jet}$ = 1.0 pc)')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.plot(spectrum_energy_ic,IC_tot,c='blue',label='Best fit values(Total)')
ax2.plot(spectrum_energy_ic,IC_cmb,c='C1',linestyle='--', label='Best fit values(CMB)')
ax2.plot(spectrum_energy_ic,IC_fir,c='C2',linestyle='--', label='Best fit values(FIR)')
#ax2.plot(spectrum_energy_ic,IC_nir,c='cyan',linestyle='--',label='Best fit values(NIR)')
ax2.errorbar(x1,y1,c='darkviolet',fmt = 'o', yerr = error_limit1, ls='None',label = 'HAWC')
ax2.errorbar(x2,y2,c='firebrick',fmt = 'D', yerr = error_limit2, ls='None',label = 'LHAASO')
ax2.text(x=0.05,y=0.05,s=r'$\chi^2/d.o.f$=4.96/6',transform=ax2.transAxes)
#左下
file3 = np.loadtxt(dir_dat1+'IC_0.5pc_spec.txt')
spectrum_energy_ic = file3[:,0]
IC_tot = file3[:,1]
IC_cmb = file3[:,2]
IC_fir = file3[:,3]
IC_nir = file3[:,4]
R_jet = ((0.5*(u.pc)).cgs).value
ax3 = fig.add_subplot(2, 2, 3) 
ax3.set_xlim(1e1,5e3)
ax3.set_ylim(3e-15,1e-10)
ax3.set_xlabel('Energy (TeV)')
ax3.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax3.set_title(r'Inverse Compton ($R_{jet}$ = 0.5 pc)')
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.plot(spectrum_energy_ic,IC_tot,c='blue',label='Best fit values(Total)')
ax3.plot(spectrum_energy_ic,IC_cmb,c='C1',linestyle='--', label='Best fit values(CMB)')
ax3.plot(spectrum_energy_ic,IC_fir,c='C2',linestyle='--', label='Best fit values(FIR)')
#ax3.plot(spectrum_energy_ic,IC_nir,c='cyan',linestyle='--',label='Best fit values(NIR)')
ax3.errorbar(x1,y1,c='darkviolet',fmt = 'o', yerr = error_limit1, ls='None',label = 'HAWC')
ax3.errorbar(x2,y2,c='firebrick',fmt = 'D', yerr = error_limit2, ls='None',label = 'LHAASO')
ax3.text(x=0.05,y=0.05,s=r'$\chi^2/d.o.f$=4.97/6', transform=ax3.transAxes)
#右下
file4 = np.loadtxt(dir_dat1+'IC_0.1pc_spec.txt')
spectrum_energy_ic = file4[:,0]
IC_tot = file4[:,1]
IC_cmb = file4[:,2]
IC_fir = file4[:,3]
IC_nir = file4[:,4]
R_jet = ((0.1*(u.pc)).cgs).value
ax4 = fig.add_subplot(2, 2, 4) 
ax4.set_xlim(1e1,5e3)
ax4.set_ylim(3e-15,1e-10)
ax4.set_xlabel('Energy (TeV)')
ax4.set_ylabel(r'Flux (erg $cm^{-2}s^{-1}$)')
ax4.set_title(r'Inverse Compton ($R_{jet}$ = 0.1 pc)')
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.plot(spectrum_energy_ic,IC_tot,c='blue',label='Best fit values(Total)')
ax4.plot(spectrum_energy_ic,IC_cmb,c='C1',linestyle='--', label='Best fit values(CMB)')
ax4.plot(spectrum_energy_ic,IC_fir,c='C2',linestyle='--', label='Best fit values(FIR)')
#ax4.plot(spectrum_energy_ic,IC_nir,c='cyan',linestyle='--',label='Best fit values(NIR)')
ax4.errorbar(x1,y1,c='darkviolet',fmt = 'o', yerr = error_limit1, ls='None',label = 'HAWC')
ax4.errorbar(x2,y2,c='firebrick',fmt = 'D', yerr = error_limit2, ls='None',label = 'LHAASO')
ax4.text(x=0.05,y=0.05,s=r'$\chi^2/d.o.f$=4.98/6', transform=ax4.transAxes)

plt.tight_layout()
plt.legend(ncol=2, frameon=False)
plt.show()
plt.savefig(dir_tar+ 'IC_all_noHESS.pdf',format="pdf", dpi=300, bbox_inches="tight")
plt.savefig(dir_tar+ 'IC_all_noHESS.png')