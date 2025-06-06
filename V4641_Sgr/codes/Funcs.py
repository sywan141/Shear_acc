import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import kv
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.special import hyp1f1

R=5
m_e = (const.m_e.cgs).value
c= (const.c.cgs).value
e = (const.e.esu).value
h = (const.h.cgs).value
sigma_sb = (const.sigma_sb.cgs).value
pi = np.pi
d = ((6.2*(u.kpc)).cgs).value
q=5/3
q=5/3
T_cmb = 2.72 # CMB temperature
z=0
xi=1
sigma_e = (8*np.pi/3)*np.square(e**2/(m_e*c**2)) 
R_jet = ((R*(u.pc)).cgs).value

def integrand(t):
        return kv(5/3, t)

def synchrotron_spectrum(B, nu, gamma):
    gamma = np.asarray(gamma)
    nu = np.asarray(nu)
    gamma_2d, nu_2d = np.meshgrid(gamma, nu, indexing='ij')
    nu_c = (3 * gamma_2d**2 * e * B) / (4 * pi * m_e * c)
    x = nu_2d / nu_c
    integral = np.vectorize(lambda xi: quad(integrand, xi, np.inf)[0])(x)
    coff = np.sqrt(3) * e**3 * B / (m_e * c**2)
    return coff * x * integral

# Calculate the total flux in a fixed energy range
# B: The magnetic field in gauss;
# gamma: energy of electrons
# n: number of spectrums
# obs_range: frequencies for X ray radiation
def Syn_flux(B, obs_range, gamma, n):
    F_matrix = synchrotron_spectrum(B, obs_range, gamma)
    # gamma axis
    F_nu = simps(n[:,None]* F_matrix, gamma, axis=0)
    # nu axis
    F_tot = simps(F_nu, obs_range)
    return F_tot / (4 * pi * d**2)

'''
# single electron spectrum
def synchrotron_spectrum(B, nu, gamma):
    nu_c = (3*gamma**2*e*B)/(4*pi*m_e*c)
    x = nu/nu_c
    integrand = lambda xi: kv(5/3, xi)
    integral, _ = quad(integrand, x, np.inf)
    coff = np.sqrt(3)*e**3*B/(m_e*c**2)
    return coff * x * integral


def Syn_flux(B,obs_range,gamma,n):
    F_tot_nu = []
    for nu in obs_range:
        F_nu_i=[]
        for gammai in gamma:
            F_nu_i.append(synchrotron_spectrum(B,nu,gammai))
        F_nu = simps(n*F_nu_i,gamma)
        F_tot_nu.append(F_nu)
    F_tot = simps(F_tot_nu,obs_range)
    return F_tot/(4*pi*d**2)
'''


def coeff_A1(D_sh,xi,Lam_max,q,B0):
    return D_sh*xi**(-1)*(Lam_max/c)**(q-1)*(m_e*c/(e*B0))**(2-q)

def coeff_A2(B0):
    U_B = np.square(B0)/8*np.pi
    U_cmb = ((4*sigma_sb)/c)*T_cmb**4*(1+z)**4     
    U_rad = U_cmb
    return (sigma_e*B0**2*(1+U_rad/U_B))/(6*np.pi*m_e*c)


def SSEPL(lgB,rate,beta,lgN,lgrho):
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
    Reg_sh = rate*R_jet
    Lam_max = rate*R_jet 
    N_tot = 10**lgN # Normalization index, 10**Norm_A
    rho = 10**lgrho
    Gamma = 1/np.sqrt(1-beta**2) # should be averaged through the jet?
    Gamma_j4 = (rate/(2*beta))*(np.log((1+beta)/(1-beta))+(2*beta)/(1-beta**2))-rate**2/(1-beta**2)    # averaged bulk lorentz factor
    #Gamma_j4 = (1/(4*np.square(beta)))*np.square(np.log((1+beta)/(1-beta))) 
    Grad=beta/Reg_sh
    lgE_min = 6

    # calculate the maximum energy for stochastic acc
    beta_Alf = (B0/np.sqrt(4*pi*((rho*c**2)))) # Alfven velocity

    if(np.isnan(beta_Alf)==True):
        return np.zeros(1)-np.inf

    Gamma_Alf = 1/np.sqrt(1-beta_Alf**2)
    gamma_st = ((15*(q+2)/(2*(6-q)))*(Gamma_Alf**2*(np.square(xi*Gamma_Alf*beta_Alf))/(Lam_max**(2*q-2)*Gamma_j4*np.square(Grad))))**(1/(4-2*q))*(e*B0/(m_e*np.square(c)))
    E_st = (((gamma_st*m_e*c**2)*(u.erg)).to(u.eV)).value
    lgE_st = np.log10(E_st)
    
    # The part is for the shearing spectrum
    D_sh = (2/15)*Gamma_j4*np.square(Grad*c)
    # shearing coefficients
    A1 = coeff_A1(D_sh,xi,Lam_max,q,B0) # under the resonance limit
    A2 = coeff_A2(B0)

    gamma_max1 = (((6-q)/2)*(A1/A2))**(1/(q-1)) # maximum energy via radiative cooling
    gamma_max2 = e*B0*R_jet/(m_e*c**2) # Hillas
    gamma_max3 = (xi*R_jet*Lam_max**(1-q))**(1/(2-q))*(e*B0/(m_e*c**2)) # MFP
    gamma_cut = np.min((gamma_max1,gamma_max2,gamma_max3))
                       
    E_cut=(((gamma_cut*(m_e*c**2))*(u.erg)).to(u.eV)).value
    lgE_cut = np.log10(E_cut)
    z_cut=((6-q)/(1-q))*(gamma_cut/gamma_max1)**(q-1) # parametre z where n becomes 0
    #w = 40/np.square(np.log((1+beta)/(1-beta)))
    w = 10*rate**2*beta**(-2)*Gamma_j4**(-1)  # no averaged escaping
    s1 = (q-1)/2+np.sqrt((5-q)**2/4+w)
    s2 = (q-1)/2-np.sqrt((5-q)**2/4+w)
    a1 = (2+s1)/(q-1)
    a2 = (2+s2)/(q-1)
    b1 = (2*s1)/(q-1)
    b2 = (2*s2)/(q-1)
    C2 = 1
    C1 = -C2*gamma_cut**(s2-s1)*(hyp1f1(a2,b2,z_cut)/hyp1f1(a1,b1,z_cut))
  
    # electron spec for shearing acc, the spectrum should begin at lower energies to ensure the number of particles
    if(lgE_st<=lgE_min) or (lgE_st>=lgE_cut):
        return np.zeros(1)-np.inf
    
    E_sh = (np.logspace(lgE_st,lgE_cut,1000)*(u.eV)).to(u.erg) # selected energy range(TeV)
    gamma_sh = (E_sh.value)/(m_e*np.square(c)) 
    mask = np.where(gamma_sh<=gamma_cut) # cutoff energy
    gamma_sh = gamma_sh[mask]
    z=((6-q)/(1-q))*(gamma_sh/gamma_max1)**(q-1) # parametre z array
    n = C1*gamma_sh**s1*hyp1f1(a1,b1,z) + C2*gamma_sh**s2*hyp1f1(a2,b2,z)

    # abandon negative and infinite values
    mask1 = np.where(n>0)
    n=n[mask1]
    gamma_sh = gamma_sh[mask1]
    mask2 = np.where(np.isfinite(n)==True)
    n=n[mask2]
    gamma_sh = gamma_sh[mask2]
    if (len(n)<=5):
        return np.zeros(1)-np.inf
    
    # electron spec for stochastic acc
    E_low = (np.logspace(lgE_min,lgE_st,100)*(u.eV)).to(u.erg)
    gamma_low = (E_low.value)/(m_e*np.square(c))
    N0 = n[0]/(gamma_low[-1])**(1-q)
    n_low = N0*gamma_low**(1-q)

    # Connect the spectrum and do the normalization
    gamma_all = np.append(gamma_low[:-1],gamma_sh)
    n_all = np.append(n_low[:-1],n)
    length = len(n_all)
    if (length<=5):
        return np.zeros(1)-np.inf

    ene_eV = ((gamma_all*m_e*c**2)*(u.erg)).to(u.eV) # energy in eV
    n_eV = n_all/((m_e*c**2)*(u.erg)).to(u.eV) # number density(1/eV)

    N_A = simps(n_all, gamma_all)
    n_all=n_all/N_A
    n_eV = n_eV/N_A
    return gamma_all, N_tot*n_all
    

[lgB,rate, beta,lgN,lgrho] =[-6.063336716277496485e-01,5.704555120271870283e-01,8.162733445002632315e-01,4.722254768247667300e+01,-32.65] 
gamma_all= SSEPL(lgB,rate,beta,lgN,lgrho)[0]
n= SSEPL(lgB,rate,beta,lgN,lgrho)[1]
obs_range = (((np.linspace(2e3,1e4,len(gamma_all))*(u.eV)).to(u.erg))).value/h   #2-10 keV
B0=(((10**lgB)*(u.uG)).to(u.G)).value
F = Syn_flux(B0,obs_range,gamma_all,n)
print(F)
#表面亮度 R_jet = ((5*(u.pc)).cgs).value, D_jet = ((100*(u.pc)).cgs).value
d_pc = ((6.2*(u.kpc)).to(u.pc)).value
arcmin_2 = (2*5/(d_pc))*3437.7467707849396**2*(100/d_pc)
print(arcmin_2)
S_B = F/arcmin_2
print('Surface Brightness(arcmin^-2):%s'%S_B)