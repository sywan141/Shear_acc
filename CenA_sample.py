# Centaurus A sample
import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from naima.models import TableModel
from naima.models import InverseCompton,Synchrotron


m_e = (const.m_e.cgs).value
c= (const.c.cgs).value
e = (const.e.esu).value
sigma_sb = (const.sigma_sb.cgs).value

# analytical solution. All paras in gaussian
def profile_sh(q,xi,B0,Lam_max,beta_j0,r_j,sh_ratio,z):
    
    B0 = (B0.to(u.G)).value
    Lam_max = (Lam_max.cgs).value
    r_j = (r_j.cgs).value

    #r_L = (gamma_r*m_e*c**2/(e*B0)) # Lamor radius in cm
    Gamma_j4 = (1/(4*np.square(beta_j0)))*np.square(np.log((1+beta_j0)/(1-beta_j0))) # bulk lorentz factor
    Delta_beta = beta_j0/(sh_ratio*r_j)
    D_sh = (1/15)*Gamma_j4*np.square(c)*np.square(Delta_beta)
    sigma_e = (8*np.pi/3)*np.square(e**2/(m_e*c**2)) # cross section of electrons
    T_cmb = 2.72 # CMB temperature
    U_B = np.square(B0)/8*np.pi
    U_cmb = ((4*sigma_sb)/c)*T_cmb**4*(1+z)**4 # only CMB is considered
    U_rad = U_cmb

    A1 = D_sh*xi**(-1)*(Lam_max/c)**(q-1)*(m_e*c/(e*B0))**(2-q)
    A2 = (sigma_e*B0**2*(1+U_rad/U_B))/(6*np.pi*m_e*c)
    gamma_max = (((6-q)/2)*(A1/A2))**(1/(q-1))
    gamma_cut = 10*gamma_max
    z_cut=((6-q)/(1-q))*(gamma_cut/gamma_max)**(q-1)
    #print(gamma_max)

    w = 80/np.square(np.log((1+beta_j0)/(1-beta_j0)))
    s1 = (q-1)/2+np.sqrt((5-q)**2/4+w)
    s2 = (q-1)/2-np.sqrt((5-q)**2/4+w)
    a1 = (2+s1)/(q-1)
    a2 = (2+s2)/(q-1)
    b1 = (2*s1)/(q-1)
    b2 = (2*s2)/(q-1)
    # n(10γ_max)=0
    C2 = 1
    C1 = -C2*gamma_cut**(s2-s1)*(hyp1f1(a2,b2,z_cut)/hyp1f1(a1,b1,z_cut))
    E_all = (np.logspace(12,18,600)*(u.eV)).to(u.erg)
    gamma_all = (E_all.value)/(m_e*np.square(c)) #10**np.arange(min,max,0.1)
    mask = np.where(gamma_all<=gamma_cut)
    gamma_all = gamma_all[mask]
    #ene_all = ((gamma_all*(m_e*np.square(c))*(u.erg)).to(u.eV)).value # from γ to E(eV)
    #ene_val = ((ene_all*(u.eV)).to(u.erg)).value # from E(eV) to E(erg)
    z=((6-q)/(1-q))*(gamma_all/gamma_max)**(q-1)
    n = C1*gamma_all**s1*hyp1f1(a1,b1,z)+C2*gamma_all**s2*hyp1f1(a2,b2,z)
    #print(n)

    return gamma_all, n

# electron SED at low energies
def profile_low(K1,alpha):
    def EPL(E):
        EPL = K1*(E)**(-alpha)
        return(EPL)
    return(EPL)

# Sample
B0 = 17.1*(u.uG)
Lam_max = 1e18*(u.cm)
r_j = 0.03*(u.kpc)
sh_ratio = 0.1
beta_j0  = 0.67
alpha = -2.31

gamma_all, n_gamma = profile_sh(q=5/3,xi = 1,B0=B0,Lam_max=Lam_max,beta_j0=beta_j0,r_j=r_j,sh_ratio=sh_ratio,z=2)
ene_eV = ((gamma_all*m_e*c**2)*(u.erg)).to(u.eV)
n_eV = n_gamma/((m_e*c**2)*(u.erg)).to(u.eV)
#print(n_eV)
K1 = n_eV[0]/ene_eV[0]**(-alpha)

ene_low = np.logspace(9,12,300)*(u.eV)
EPL = profile_low(K1,alpha)(ene_low)
ene_all = np.append(ene_low[:-1],ene_eV)
n_all = np.append(EPL[:-1],n_eV)
plt.title('EPL for electrons')
plt.loglog(ene_all,n_all)
plt.xlabel('E(eV)')
plt.ylabel('N(E)(1/eV)')
plt.show()

# Calculate syn
spectrum_energy = np.logspace(-7, 18, 1000)
spectrum_energy = spectrum_energy*(u.eV)
EC = TableModel(ene_all,n_all)
SYN = Synchrotron(EC, B=B0)
sed_SYN = SYN.sed(spectrum_energy)
plt.title('Sync')
plt.loglog(spectrum_energy,sed_SYN)
plt.xlabel('E(eV)')
plt.ylabel(r'$E^2(dN/dE) [erg \cdot s^{-1}cm^{-2}]$')
plt.xlim(1e-6,1e6)
plt.ylim(1e-100)
plt.show()

# Calculate IC-CMB
IC = InverseCompton(EC,seed_photon_fields=['CMB'])
sed_IC = IC.sed(spectrum_energy) # distance=1.5 * u.kpc
#print(sed_IC)
plt.title('IC-CMB')
plt.loglog(spectrum_energy,sed_IC)
plt.xlabel('E(eV)')
plt.ylabel(r'$E^2(dN/dE) [erg \cdot s^{-1}cm^{-2}]$')
plt.xlim(1e0,1e15)
plt.ylim(1e-100)
plt.show()

# Show both in the same plot
plt.title('Syn-IC')
plt.loglog(spectrum_energy,sed_SYN,label='Synchrotron')
plt.loglog(spectrum_energy,sed_IC,label = 'IC-CMB')
plt.xlabel('E(eV)')
plt.ylabel(r'$E^2(dN/dE) [erg \cdot s^{-1}cm^{-2}]$')
plt.xlim(1e-6)
plt.ylim(1e-100)
plt.legend()
plt.show()