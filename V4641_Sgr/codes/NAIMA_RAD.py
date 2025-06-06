import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from naima.models import TableModel
from naima.models import InverseCompton,Synchrotron
from scipy.integrate import simps
import math

m_e = (const.m_e.cgs).value
c= (const.c.cgs).value
e = (const.e.esu).value
sigma_sb = (const.sigma_sb.cgs).value
sigma_e = (8*np.pi/3)*np.square(e**2/(m_e*c**2)) 
m_p = (const.m_p.cgs).value
D_jet = ((100*(u.pc)).cgs).value
pi = np.pi
inf = np.inf
nan = float('nan')
q=5/3
T_cmb = 2.72 # CMB temperature
z=0
xi=1
q_sa = 2

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
err2 = (upper2+lower2)/2
error_limit2 = [lower2,upper2]
a2 = np.column_stack((x2,y2,err2))
# HESS, in erg
x3 = (file3[:,0]*(u.TeV)).value
y3 = ((file3[:,1]*(u.erg))).value
upper3 = (file3[:,2]*(u.erg)).value-y3
lower3 = y3-(file3[:,3]*(u.erg)).value
err3 = (upper3+lower3)/2
error_limit3 = [lower3,upper3]
a3 = np.column_stack((x3,y3,err3))
a_all = np.row_stack((a1,a2,a3))
sorted_a_all = a_all[np.argsort(a_all[:, 0])]
ene_obs_data = np.float64(sorted_a_all[:,0])

# Galactic extinction
ene_syn = np.logspace(1,9,500) # X ray
#ene_obs = np.logspace(0,3,500) # Gamma ray
#ene_obs = np.logspace(1,4,2000) # Gamma ray
ene_obs = np.logspace(-3.1,3.2,1000) # extended gamma ray

x1 = ene_obs_data # 17 data points
y1 = np.loadtxt('/home/wsy/V4641_Sgr/Extinction_factors.txt')

x2 = np.loadtxt('/home/wsy/V4641_Sgr/PeV_factors.txt')[:,0]
y2 = np.loadtxt('/home/wsy/V4641_Sgr/PeV_factors.txt')[:,1]
#factors = np.interp(ene_obs, x1, y1, left=1, right=0.5081732)
#factors = np.interp(ene_obs, x2, y2)
factors = np.interp(ene_obs, x2, y2, left=1) # extended spectrum

# Combined spectrum model of stochastic and shear acceleration
def coeff_A1(D_sh,xi,Lam_max,q,B0):
    return D_sh*xi**(-1)*(Lam_max/c)**(q-1)*(m_e*c/(e*B0))**(2-q)

def coeff_A2(B0):
    U_B = np.square(B0)/(8*np.pi)
    U_cmb = ((4*sigma_sb)/c)*T_cmb**4*(1+z)**4     
    U_rad = U_cmb
    return (sigma_e*B0**2*(1+U_rad/U_B))/(6*np.pi*m_e*c)

def Elect_spec(R,lgB,rate,beta,lgN,lgrho):
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
    R_jet = ((R*(u.pc)).cgs).value
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
    E_low = (np.logspace(lgE_min,lgE_st,1000)*(u.eV)).to(u.erg)
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
   
    return ene_eV, n_eV

def NAIMA_IC(ene_eV,n_eV,beta,lgN):
    N_tot=10**lgN
    Gamma=1/np.sqrt(1-beta**2)
    # calculate
    EC = TableModel(ene_eV,n_eV)
    IC0 = InverseCompton(EC,['CMB','FIR','NIR'],Eemax = 1e20*(u.eV))
    IC1 = InverseCompton(EC,['CMB'],Eemax = 1e20*(u.eV))
    IC2 = InverseCompton(EC,['FIR'],Eemax = 1e20*(u.eV))
    IC3 = InverseCompton(EC,['NIR'],Eemax = 1e20*(u.eV))
    spectrum_energy_ic = ene_obs * (u.TeV)
    sed_IC0 = IC0.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # Total
    sed_IC1 = IC1.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # CMB
    sed_IC2 = IC2.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # FIR
    sed_IC3 = IC3.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # NIR
    sed_IC_data = IC0.sed(ene_obs*(u.TeV), distance=6.2 * u.kpc) # observed, used to calculate chi2
    sed_IC0=N_tot*(sed_IC0.value)
    sed_IC1=N_tot*(sed_IC1.value)
    sed_IC2=N_tot*(sed_IC2.value)
    sed_IC3=N_tot*(sed_IC3.value)
    sed_IC_data=N_tot*(sed_IC_data.value)

    rad = (72/360)*2*pi # The angle between the jet axis and the line of sight
    D = 1/(Gamma**4*(1-beta*math.cos(rad))**3)
    
    sed_IC0 = factors*D*sed_IC0
    sed_IC1 = factors*D*sed_IC1
    sed_IC2 = factors*D*sed_IC2
    sed_IC3 = factors*D*sed_IC3 
    
    sed_IC_data = factors*D*sed_IC_data  # Take the jet beaming effect into consideration

    return sed_IC0, sed_IC1, sed_IC2, sed_IC3, sed_IC_data

def NAIMA_SYN(ene_eV,n_eV,lgB,beta,lgN):
    # calculate
    Gamma = 1/np.sqrt(1-beta**2)
    N_tot=10**lgN

    # synchrotron emission
    EC = TableModel(ene_eV,n_eV)
    SYN = Synchrotron(EC, B=(10**lgB)* u.uG)
    spectrum_energy_syn = ene_syn * (u.eV)
    sed_SYN = SYN.sed(spectrum_energy_syn, distance=6.2 * u.kpc)
    sed_SYN=N_tot*(sed_SYN.value)
    rad = (72/360)*2*pi # The angle between the jet axis and the line of sight
    D = 1/(Gamma**4*(1-beta*math.cos(rad))**3)
    sed_SYN = D*sed_SYN # Take the jet beaming effect into consideration

    return sed_SYN


def coeff_A3(R_jet, D_sh,A1):
    return (3*R_jet**2*D_sh)/(2*c**2*A1)

def para_z(A1,A2,gamma,q):
    gamma_max_1 = (((6-q)/2)*(A1/A2))**(1/(q-1))
    z=((6-q)/(1-q))*(gamma/gamma_max_1)**(q-1)
    return z

def F1(w,q,gamma,z):
    s1 = (q-1)/2+np.sqrt((5-q)**2/4+w)
    a1 = (2+s1)/(q-1)
    b1 = (2*s1)/(q-1)
    return gamma**s1*hyp1f1(a1,b1,z)

def F2(w,q,gamma,z):
    s2 = (q-1)/2-np.sqrt((5-q)**2/4+w)
    a2 = (2+s2)/(q-1)
    b2 = (2*s2)/(q-1)
    return gamma**s2*hyp1f1(a2,b2,z)

def SSEPL(R,lgB,rate,beta,lgK1,lgrho):
    K1=10**lgK1
    q_rsn = 0
#------------------------------------------------------------------------------------------------------------------
    # free and fixed parameters
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
    R_jet = ((R*(u.pc)).cgs).value
    Reg_sh = rate*R_jet
    Lam_max = Reg_sh
    beta = beta
    rho = 10**lgrho

    Gamma = 1/np.sqrt(1-beta**2) # should be averaged through the jet?
    Gamma_j4 = (rate/(2*beta))*(np.log((1+beta)/(1-beta))+(2*beta)/(1-beta**2))-rate**2/(1-beta**2)
    Grad=beta/Reg_sh
    D_sh = (2/15)*Gamma_j4*np.square(Grad*c)
    beta_Alf = (B0/np.sqrt(4*pi*((rho*c**2)))) # Alfven velocity
    lgE_min = 6
    gamma_min=(((10**(lgE_min))*(u.eV)).to(u.erg)).value/(m_e*c**2)
#------------------------------------------------------------------------------------------------------------------

    # calculate the maximum energy for stochastic acc

    if(np.isnan(beta_Alf)==True):
        return np.zeros(1)-np.inf

    # injection
    Gamma_Alf = 1/np.sqrt(1-beta_Alf**2)
    gamma_inj = ((15*(q+2)/(2*(6-q)))*(Gamma_Alf**2*(np.square(xi*Gamma_Alf*beta_Alf))/(Lam_max**(2*q-2)*Gamma_j4*np.square(Grad))))**(1/(4-2*q))*(e*B0/(m_e*np.square(c)))
    
    # shearing coefficients
    A1 = coeff_A1(D_sh,xi,Lam_max,q,B0) # under the resonance limit
    A2 = coeff_A2(B0) 
    #A1_rsn = coeff_A1(D_sh,xi,Lam_max,q_rsn,B0)
    #A3_rsn = coeff_A3(D_sh,A1_rsn)

    # resonance limit
    gamma_rsn = e*B0*Lam_max/(m_e*c**2)
    
    # cutoff
    gamma_cut =  (e*B0/(m_e*c**2))*np.sqrt(xi*Lam_max*R_jet)  # MFP when timescale is changed
    #gamma_cut = e*B0*R_jet/(m_e*c**2)                                 

    # spectrum after gamma_rsn
    gamma_gyr=np.logspace(np.log10(gamma_rsn),np.log10(gamma_cut),1000)
    b_s = 1-q_rsn
    w = 10*rate**2/(beta**2*Gamma_j4)
    c_s = 2*q_rsn-6-w
    p_s = (-b_s+np.sqrt(b_s**2-4*c_s))/2
    q_s = (-b_s-np.sqrt(b_s**2-4*c_s))/2
    D2 = K1
    D1 = -D2*(10**np.log10(gamma_cut))**(q_s-p_s)
    n_gyr = D1*gamma_gyr**p_s + D2*gamma_gyr**q_s
    #print(n_gyr[-20:])

    #------------------------------------------------------------------------------------------------------------
    # spectrum between gamma_inj and gamma_gyr
    gamma_1= np.logspace(np.log10(gamma_inj),np.log10(gamma_rsn),1000)
    z_1= para_z(A1,A2,gamma_1,q)
    

    z_cut_1 = para_z(A1,A2,gamma_1[-1],q)
    z_cut_2 = para_z(A1,A2,gamma_1[-2],q)
    z_cut_3 = para_z(A1,A2,gamma_1[-3],q)

    F1_1 = F1(w,q,gamma_1[-1],z_cut_1)
    F1_2 = F1(w,q,gamma_1[-2],z_cut_2)
    F1_3 = F1(w,q,gamma_1[-3],z_cut_3)
    F2_1 = F2(w,q,gamma_1[-1],z_cut_1)
    F2_2 = F2(w,q,gamma_1[-2],z_cut_2)
    F2_3 = F2(w,q,gamma_1[-3],z_cut_3)
    cont = (F2_3-2*F2_2+F2_1)/(F1_3-2*F1_2+F1_1)

    C2= n_gyr[0]/(F2_1-cont*F1_1)
    C1 = -cont*C2 #(n_gyr[0]-C2*F2(w,q,gamma_rsn,z_cut_1))/F1(w,q,gamma_rsn,z_cut_1)
    n_1 = C1*F1(w,q,gamma_1,z_1)+C2*F2(w,q,gamma_1,z_1)
    #------------------------------------------------------------------------------------------------------------
    # spectrum between gamma_min and gamma_inj
    gamma_0=np.logspace(np.log10(gamma_min),np.log10(gamma_inj),1000)
    N0 = n_1[0]/(gamma_inj)**(1-q)
    n_0 = N0*gamma_0**(1-q)
    #print(C1)

    #------------------------------------------------------------------------------------------------------------
    # connect the spectrum
    gamma_intv = np.append(gamma_0[:-1],gamma_1[:-1])
    gamma_all = np.append(gamma_intv,gamma_gyr)
    n_intv = np.append(n_0[:-1],n_1[:-1])
    n_all = np.append(n_intv,n_gyr)
    n_all = n_all.astype(float)

    # electron spec for shearing acc, the spectrum should begin at lower energies to ensure the number of particles
    if(gamma_inj<=gamma_min) or (gamma_rsn<=gamma_inj):
        return np.zeros(1)-np.inf
    
    # abandon negative and infinite values
    mask1 = np.where(n_all>0)
    n_all=n_all[mask1]
    gamma_all = gamma_all[mask1]
    mask2 = np.where(np.isfinite(n_all)==True)
    n_all=n_all[mask2]
    gamma_all = gamma_all[mask2]
    if (len(n_all)<=5):
        return np.zeros(1)-np.inf


    ene_eV = ((gamma_all*m_e*c**2)*(u.erg)).to(u.eV) # energy in eV
    n_eV = n_all/((m_e*c**2)*(u.erg)).to(u.eV) # number density(1/eV)

    
    return ene_eV,n_eV


def IC_SA(R,lgB,rate,lgK,lgrho):
    R_jet = ((R*(u.pc)).cgs).value
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
    Lam_max = rate*R_jet
    K_all = 10**lgK # Normalization index, 10**Norm_A
    rho = 10**lgrho
    lgE_min = 6
    gamma_min = ((((10**lgE_min)*(u.eV)).to(u.erg))/(m_e*c**2)).value
    lg_min = np.log10(gamma_min)

    # calculate the coefficients for stochastic acc
    beta_Alf = (B0/np.sqrt(4*pi*((rho*c**2)))) # Alfven velocity +B0**2/(4*pi)
    Gamma_Alf = 1/np.sqrt(1-beta_Alf**2)
    U_B = np.square(B0)/(8*np.pi)
    U_cmb = ((4*sigma_sb)/c)*T_cmb**4*(1+z)**4 # only CMB is considered
    U_rad = U_cmb
    A1 = ((xi*Gamma_Alf**4*beta_Alf**2*c)/(Lam_max)**(q_sa-1))*(m_e*c**2/(e*B0))**(q_sa-2)
    A2 = (sigma_e*B0**2*(1+U_rad/U_B))/(6*np.pi*m_e*c)
    A3 = ((3*R_jet**2*xi*Lam_max**(1-q_sa))/(2*c))*((e*B0)/(m_e*c**2))**(2-q_sa)
    #print(beta_Alf)
    
    # This part is for the shearing spectrum
    gamma_max1 = ((2*A2)/((2+q_sa)*A1))**(1/(q_sa-3)) # cooling limit
    gamma_max2 = e*B0*R_jet/(m_e*c**2) # hillas condition
    #gamma_cut = np.min((gamma_max1,gamma_max2))
    #print(gamma_max1/gamma_max2)
    gamma_cut = gamma_max2
    lg_cut = np.log10(gamma_cut)
    gamma_all = np.logspace(lg_min,lg_cut,500)
    z_cut = -((2*A2)/A1)*gamma_cut

    s1 = 1/2+np.sqrt(9/4+2/(A1*A3))
    s2 = 1/2-np.sqrt(9/4+2/(A1*A3))
    a1 = 1+s1
    a2 = 1+s2
    b1 = 2*s1
    b2 = 2*s2
    C2 = 1
    C1 = -C2*gamma_cut**(s2-s1)*(hyp1f1(a2,b2,z_cut)/hyp1f1(a1,b1,z_cut))
  
    z_g = -((2*A2)/A1)*gamma_all
    n = C1*gamma_all**s1*hyp1f1(a1,b1,z_g) + C2*gamma_all**s2*hyp1f1(a2,b2,z_g)
    n = np.array(n, dtype=np.float64)

    # abandon negative and infinite values
    mask1 = np.where(n>0)
    n=n[mask1]
    gamma_all = gamma_all[mask1]
    mask2 = np.where(np.isfinite(n)==True)
    n=n[mask2]
    gamma_all = gamma_all[mask2]
    

    ene_eV = ((gamma_all*m_e*c**2)*(u.erg)).to(u.eV) # energy in eV
    n_eV = n/((m_e*c**2)*(u.erg)).to(u.eV) # number density(1/eV)
    
    EC = TableModel(ene_eV,n_eV)
    IC0 = InverseCompton(EC,['CMB','FIR','NIR'],Eemax = 1e20*(u.eV))
    IC1 = InverseCompton(EC,['CMB'],Eemax = 1e20*(u.eV))
    IC2 = InverseCompton(EC,['FIR'],Eemax = 1e20*(u.eV))
    IC3 = InverseCompton(EC,['NIR'],Eemax = 1e20*(u.eV))
    spectrum_energy_ic = ene_obs * (u.TeV)
    sed_IC0 = IC0.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # Total
    sed_IC1 = IC1.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # CMB
    sed_IC2 = IC2.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # FIR
    sed_IC3 = IC3.sed(spectrum_energy_ic, distance=6.2 * u.kpc) # NIR
    
    sed_IC0=K_all*factors*(sed_IC0.value)
    sed_IC1=K_all*factors*(sed_IC1.value)
    sed_IC2=K_all*factors*(sed_IC2.value)
    sed_IC3=K_all*factors*(sed_IC3.value)

    return sed_IC0, sed_IC1, sed_IC2, sed_IC3