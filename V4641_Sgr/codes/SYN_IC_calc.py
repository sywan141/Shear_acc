import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from naima.models import TableModel
from naima.models import Synchrotron,InverseCompton
from scipy.integrate import simpson
import math
import warnings
warnings.filterwarnings("ignore")
#from multiprocessing import Pool
from scipy.integrate import simps

R_pc = 0.5
m_e = (const.m_e.cgs).value
c= (const.c.cgs).value
e = (const.e.esu).value
sigma_sb = (const.sigma_sb.cgs).value
m_p = (const.m_p.cgs).value
R_jet = ((R_pc*(u.pc)).cgs).value
D_jet = ((100*(u.pc)).cgs).value
pi = np.pi
inf = np.inf
nan = float('nan')
E_pm = ((0.8*(u.PeV)).to(u.erg)).value
gamma_pm=E_pm/(m_e*c**2)

#-------------------------------------------------------------------------------------------------------------------------------


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

ene_obs = np.float64(sorted_a_all[:,0])
flux_obs = np.float64(sorted_a_all[:,1])
err_obs = np.float64(sorted_a_all[:,2])

# Galactic extinction
x = ene_obs
y = np.loadtxt('/home/wsy/V4641_Sgr/Extinction_factors.txt')
ene_tau0 = np.logspace(0,3,1000) # TeV energies
ene_tau1 = ene_obs
factors = np.interp(ene_tau0, x, y, left=1, right=0.5081732)
factors_17 = np.interp(ene_obs, x, y, left=1, right=0.5081732)


#-------------------------------------------------------------------------------------------------------------------------------

# Best fits with extinctions
#datadir = '/home/wsy/V4641_Sgr/results/paras/Shear_Ratio_Lammax=Reg_DIFFUSION_MODIFIED_52_GalacticExtinction_q=1.6666666666666667_widerange/'
#filename1 = 'paras_lgB0=0.39_rate=0.1_beta=0.4_lgN=50.97451657_lgrho=-30.2_n=100000_weighted.txt'
#filename2 = 'chi2_lgB0=0.39_rate=0.1_beta=0.4_lgN=50.97451657_lgrho=-30.2_n=100000_weighted.txt'

datadir = '/home/wsy/V4641_Sgr/results/paras/MFP_xi=1_0.5pc_limited_DIFFUSION_MODIFIED_5_GalacticExtinction_q=1.6666666666666667_widerange/'
filename1 = 'paras_lgB0=-0.3_rate=0.2_beta=0.5851197_lgN=47.08667333_lgrho=-32.0421692_n=60000_weighted.txt'
filename2 = 'chi2_lgB0=-0.3_rate=0.2_beta=0.5851197_lgN=47.08667333_lgrho=-32.0421692_n=60000_weighted.txt'

file1 = np.loadtxt(datadir+filename1)
file2 = np.loadtxt(datadir+filename2)
mask0 = np.argmax(file2) # maximum chi2 likelyhood
theta = file1[mask0]
print(-2*file2[mask0])
print(file1[mask0])
para1 = file1[:,0]
chi2_prob = file2[:]


#mask1 =(para1>=0.25-0.32) 
#mask2 = (para1<=0.25+0.47)

#mask3 =(chi2_prob>np.max(chi2_prob-1)) 

mask1 =(para1>=-0.151-0.488) 
mask2 = (para1<=-0.151+0.511)

mask = mask1 & mask2
para_list1=file1[mask]
para_list1=para_list1[np.argsort(para_list1[:,0])]
#print(para_list1[0])
length = len(para_list1)
#-------------------------------------------------------------------------------------------------------------------------------
mask_cut = np.linspace(0,length-1,num=100)
int_cut = []
for num in mask_cut: 
    num=int(num)
    int_cut.append(num)
para_cut1 = para_list1[int_cut,:]
#-------------------------------------------------------------------------------------------------------------------------------

# return the electron spectrum, 
def Elect_spec(lgB,rate,beta,lgN,lgrho):
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
    Reg_sh = rate*R_jet
    Lam_max = Reg_sh
    eta_sh=rate
    beta = beta
    N_tot = 10**lgN # Normalization index, 10**Norm_A
    rho = 10**lgrho
    #R_jet = ((5*(u.pc)).cgs).value  # ~ 0.1*R_jet
    #Lam_max = (eta_lam*R_jet*(u.cm)).value
    z=0
    xi=1
    q=5/3
    Delta_r = Reg_sh
    Gamma = 1/np.sqrt(1-beta**2) # should be averaged through the jet?
    #intE_jet = rho*(Gamma-1)*c**2
    #P_jet = ((1+Gamma)/(3*Gamma))*intE_jet
    Gamma_j4 = (eta_sh/(2*beta))*(np.log((1+beta)/(1-beta))+(2*beta)/(1-beta**2))-eta_sh**2/(1-beta**2) # bulk Lorentz factor
    Grad=beta/Delta_r
    lgE_min = 6

    # calculate the maximum energy for stochastic acc
    beta_Alf = (B0/np.sqrt(4*pi*((rho*c**2)))) # Alfven velocity

    if(np.isnan(beta_Alf)==True):
        return np.zeros(1)-np.inf

    Gamma_Alf = 1/np.sqrt(1-beta_Alf**2)
    gamma_st = ((15*(q+2)/(2*(6-q)))*(Gamma_Alf**2*(np.square(xi*Gamma_Alf*beta_Alf))/(Lam_max**(2*q-2)*Gamma_j4*np.square(Grad))))**(1/(4-2*q))*(e*B0/(m_e*np.square(c)))
    E_st = (((gamma_st*m_e*c**2)*(u.erg)).to(u.eV)).value
    lgE_st = np.log10(E_st)
    #print(beta_Alf)
    #print(lgE_st)
    
    # This part is for the shearing spectrum
    D_sh = (2/15)*Gamma_j4*np.square(Grad*c)
    sigma_e = (8*np.pi/3)*np.square(e**2/(m_e*c**2)) # cross section of electrons
    T_cmb = 2.72 # CMB temperature
    U_B = np.square(B0)/(8*np.pi)
    U_cmb = ((4*sigma_sb)/c)*T_cmb**4*(1+z)**4 # only CMB is considered
    U_rad = U_cmb
    A1 = D_sh*xi**(-1)*(Lam_max/c)**(q-1)*(m_e*c/(e*B0))**(2-q)
    A2 = (sigma_e*B0**2*(1+U_rad/U_B))/(6*np.pi*m_e*c)
    gamma_max1 = (((6-q)/2)*(A1/A2))**(1/(q-1))
    #gamma_max2 = e*B0*Lam_max/(m_e*c**2)
    gamma_max2 = (xi*R_jet*Lam_max**(1-q))**(1/(2-q))*(e*B0/(m_e*c**2)) # MFP
    gamma_max = np.min((gamma_max1,gamma_max2))
    gamma_cut = gamma_max # n(γ_max)=0
    E_cut=(((gamma_cut*(m_e*c**2))*(u.erg)).to(u.eV)).value
    lgE_cut = np.log10(E_cut)
    #print(lgE_cut)
    z_cut=((6-q)/(1-q))*(gamma_cut/gamma_max)**(q-1)
    #w = 40/np.square(np.log((1+beta)/(1-beta)))
    w = 10*eta_sh**2*beta**(-2)*Gamma_j4**(-1)
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
    z=((6-q)/(1-q))*(gamma_sh/gamma_max)**(q-1)
    n = C1*gamma_sh**s1*hyp1f1(a1,b1,z)+C2*gamma_sh**s2*hyp1f1(a2,b2,z)

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

    N_A = simpson(n_all, gamma_all)
    #print(n_all)
    #print(N_A)
    #print(gamma_all)
    n_all=n_all/N_A
    n_eV = n_eV/N_A

    return ene_eV, n_eV

def NAIMA_SYN(ene_eV,n_eV,lgB,beta,lgN):
    # calculate
    Gamma = 1/np.sqrt(1-beta**2)
    N_tot=10**lgN

    # synchrotron emission
    ene_obs = np.logspace(1,5,1000)
    EC = TableModel(ene_eV,n_eV)
    SYN = Synchrotron(EC, B=(10**lgB)* u.uG)
    spectrum_energy_syn = ene_obs * (u.eV)
    sed_SYN = SYN.sed(spectrum_energy_syn, distance=6.2 * u.kpc)
    sed_SYN=N_tot*(sed_SYN.value)
    rad = (72/360)*2*pi # The angle between the jet axis and the line of sight
    D = 1/(Gamma**4*(1-beta*math.cos(rad))**3)
    sed_SYN = D*sed_SYN # Take the jet beaming effect into consideration

    return sed_SYN
#-------------------------------------------------------------------------------------------------------------------------------

def NAIMA_IC(ene_eV,n_eV,beta,lgN):
    N_tot=10**lgN
    Gamma=1/np.sqrt(1-beta**2)
    # calculate
    ene_obs_ic = np.logspace(0,3,1000)
    EC = TableModel(ene_eV,n_eV)
    IC0 = InverseCompton(EC,['CMB','FIR','NIR'],Eemax = 1e20*(u.eV))
    IC1 = InverseCompton(EC,['CMB'],Eemax = 1e20*(u.eV))
    IC2 = InverseCompton(EC,['FIR'],Eemax = 1e20*(u.eV))
    IC3 = InverseCompton(EC,['NIR'],Eemax = 1e20*(u.eV))
    spectrum_energy_ic = ene_obs_ic * (u.TeV)
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
    
    sed_IC_data = factors_17*D*sed_IC_data  # Take the jet beaming effect into consideration

    return sed_IC0, sed_IC1, sed_IC2, sed_IC3, sed_IC_data
#-------------------------------------------------------------------------------------------------------------------------------

def Kine_L(lgrho,beta):
    #L = ((50*(u.pc)).cgs).value
    #R = ((5*(u.pc)).cgs).value
    Gamma = 1/np.sqrt(1-beta**2)
    #Kine_L = beta*c*(Gamma-1)*N_tot*m_p*c**2/L # proton mass
    Kine_L = beta*c*(Gamma-1)*(10**lgrho/m_p)*pi*R_jet**2*m_p*c**2
    return Kine_L

#-------------------------------------------------------------------------------------------------------------------------------


def Kine_L_N(lgN,beta):
    #R = ((5*(u.pc)).cgs).value
    Gamma = 1/np.sqrt(1-beta**2)
    #Kine_L = beta*c*(Gamma-1)*N_tot*m_p*c**2/L # proton mass
    rho = (10**lgN/(pi*R_jet**2*D_jet))*m_p
    Kine_L_N = beta*c*(Gamma-1)*(rho/m_p)*pi*R_jet**2*m_p*c**2
    return Kine_L_N


#-------------------------------------------------------------------------------------------------------------------------------

# synchrotron radiation
lgB_best,rate_best,beta_best,lgN_best,lgrho_best=theta
spectrum_energy_syn = np.logspace(1,5,1000)
spectrum_energy_ic = np.logspace(0,3,1000)

SYN_all=[]
IC_all=[]
plt.figure()
plt.title('V4641 Sgr')
plt.semilogx()
plt.semilogy()
plt.xlim(1e1,1e5)
plt.ylim(1e-16,1e-13)
plt.xlabel('Energy(eV)')
plt.ylabel(r'Flux(erg $cm^{-2}s^{-1}$)')
for paras in para_cut1:
    lgB,rate,beta,lgN,lgrho=paras
    ene_eV,n_eV = Elect_spec(lgB,rate,beta,lgN,lgrho)
    sed_SYN = NAIMA_SYN(ene_eV,n_eV,lgB,beta,lgN)
    SYN_all.append(sed_SYN)
    #plt.plot(spectrum_energy_syn,sed_SYN)
SYN_all=np.array(SYN_all)
sed_min = np.min(SYN_all,axis=0)
sed_max = np.max(SYN_all,axis=0)
plt.fill_between(spectrum_energy_syn, sed_min, sed_max, color="lightcoral", label=r" 100 SEDs in 1$ \sigma $")
ene_best,n_best=Elect_spec(lgB_best,rate_best,beta_best,lgN_best,lgrho_best)
plt.plot(spectrum_energy_syn,NAIMA_SYN(ene_best,n_best,lgB_best,beta_best,lgN_best),c='red',label='Best fit values')
plt.legend(frameon=False)
plt.savefig('/home/wsy/V4641_Sgr/SYN_%spc.png'%R_pc)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

# IC radiation
plt.figure()
plt.title('V4641 Sgr')
plt.semilogx()
plt.semilogy()
plt.xlim(1e0,1e3)
plt.ylim(3e-15,1e-10)
plt.xlabel('Energy(TeV)')
plt.ylabel(r'Flux(erg $cm^{-2}s^{-1}$)')
for paras in para_cut1:
    lgB,rate,beta,lgN,lgrho=paras
    ene_eV,n_eV = Elect_spec(lgB,rate,beta,lgN,lgrho)
    sed_IC = NAIMA_IC(ene_eV,n_eV,beta,lgN)[0]
    IC_all.append(sed_IC)
    #plt.plot(spectrum_energy_syn,sed_SYN)
IC_all=np.array(IC_all)
sed_min = np.min(IC_all,axis=0)
sed_max = np.max(IC_all,axis=0)
plt.fill_between(spectrum_energy_ic, sed_min, sed_max, color="lightskyblue", label=r" 100 SEDs in 1$ \sigma $")
#ene_best,n_best=Elect_spec(lgB_best,rate_best,beta_best,lgN_best,lgrho_best)
IC_tot, IC_cmb, IC_fir, IC_nir, IC_obs= NAIMA_IC(ene_best,n_best,beta_best,lgN_best)
plt.plot(spectrum_energy_ic,IC_tot,c='blue',label='Best fit values(Total)')
plt.plot(spectrum_energy_ic,IC_cmb,c='C1',linestyle='--', label='Best fit values(CMB)')
plt.plot(spectrum_energy_ic,IC_fir,c='C2',linestyle='--', label='Best fit values(FIR)')
plt.plot(spectrum_energy_ic,IC_nir,c='cyan',linestyle='--',label='Best fit values(NIR)')
plt.errorbar(x1,y1,c='darkviolet',fmt = 'o', yerr = error_limit1,ls='None',label = 'HAWC')
plt.errorbar(x2,y2,c='firebrick',fmt = 'D', yerr = error_limit2,ls='None',label = 'LHAASO')
plt.errorbar(x3,y3,c='forestgreen',fmt = 's', yerr = error_limit3,ls='None',label = 'HESS')
plt.legend(frameon=False)
plt.savefig('/home/wsy/V4641_Sgr/IC_%spc.png'%R_pc)
plt.show()


#-------------------------------------------------------------------------------------------------------------------------------

'''
# 设置图形大小和布局
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)  # 设置主图和子图

# 主图: V4641 Sgr的SED图
ax1.set_title('V4641 Sgr')
ax1.semilogx()
ax1.semilogy()
ax1.set_xlim(1e0, 1e3)
ax1.set_ylim(1e-14, 1e-10)
ax1.set_xlabel('Energy(TeV)')
ax1.set_ylabel(r'Flux(erg $cm^{-2}s^{-1}$)')

# 绘制填充区域
for paras in para_cut1:
    lgB, rate, beta, lgN, lgrho = paras
    sed_IC = ICRad(lgB, rate, beta, lgN, lgrho)[0]
    IC_all.append(sed_IC)
IC_all = np.array(IC_all)
sed_min = np.min(IC_all, axis=0)
sed_max = np.max(IC_all, axis=0)

ax1.fill_between(spectrum_energy_ic, sed_min, sed_max, color="lightskyblue", label=r" 100 SEDs in 1$ \sigma $")
ax1.plot(spectrum_energy_ic, ICRad(lgB_best, rate_best, beta_best, lgN_best, lgrho_best)[0], c='blue', label='Best fit values(Total)')
ax1.plot(spectrum_energy_ic, ICRad(lgB_best, rate_best, beta_best, lgN_best, lgrho_best)[1], c='C1', linestyle='--', label='Best fit values(CMB)')
ax1.plot(spectrum_energy_ic, ICRad(lgB_best, rate_best, beta_best, lgN_best, lgrho_best)[2], c='C2', linestyle='--', label='Best fit values(FIR)')
# plt.plot(spectrum_energy_ic, ICRad(lgB_best, rate_best, beta_best, lgN_best, lgrho_best)[3], c='cyan', label='Best fit values(NIR)')
ax1.errorbar(x1, y1, c='darkviolet', fmt='o', yerr=error_limit1, ls='None', label='HAWC')
ax1.errorbar(x2, y2, c='firebrick', fmt='D', yerr=error_limit2, ls='None', label='LHAASO')
ax1.errorbar(x3, y3, c='forestgreen', fmt='s', yerr=error_limit3, ls='None', label='HESS')

# 添加图例
ax1.legend(frameon=False)

# 创建下方小图
# 子图布局 ax2用于显示卡方值
data_x = ene_obs
chi2_values = (flux_obs-ICRad(lgB_best, rate_best, beta_best, lgN_best, lgrho_best)[4])**2/(err_obs)**2
ax2 = ax1.inset_axes([0.0, -0.2, 0.8, 0.2])  # 设置小图的位置和大小：位置（x, y），宽度和高度
ax2.set_xticks(ax1.get_xticks())  # 使用主图的x轴刻度
ax2.set_xticklabels(ax1.get_xticklabels())  # 使用主图的x轴标签

ax2.scatter(data_x, chi2_values, label=r'$\chi^2$', color='purple')  # 假设chi_squared_values是每个点的卡方值
#ax2.set_xlabel('Energy(TeV)')
ax2.set_ylabel(r'$\chi^2$')

# 调整小图的显示范围
#ax2.set_xlim([0, len(data_points) - 1])  # 根据你的数据调整
#ax2.set_ylim([0, np.max(chi_squared_values) + 5])  # 设置y轴显示的范围

# 添加卡方图例
ax2.legend()

# 保存并显示图像
plt.savefig('/home/wsy/V4641_Sgr/IC_ext_diff_40_with_chi_squared.png')
plt.show()
'''
#-------------------------------------------------------------------------------------------------------------------------------
# Luminosity
L_k = []
L_kn = []
for paras in para_cut1:
    lgB,rate,beta,lgN,lgrho=paras
    L_ki = Kine_L(lgrho,beta)
    L_kni = Kine_L_N(lgN,beta)
    L_k.append(L_ki)
    L_kn.append(L_kni)
    #plt.plot(spectrum_energy_syn,sed_SYN)
idx_n = np.linspace(0,length-1,num=100)
y1 = L_k
y2 = L_kn
mean_k = np.sum(y1)/len(y1)
mean_kn = np.sum(y2)/len(y2)

L_best_k = Kine_L(lgrho_best, beta_best)
L_best_kn = Kine_L_N(lgN_best, beta_best)

plt.figure()
plt.title('Kinetic Luminosity')
#plt.semilogx()
plt.semilogy()
plt.xlim(0,1+np.max(idx_n))
plt.ylim(1e35,5e39)
plt.xlabel('Index')
plt.ylabel('Luminosity(erg/s)')
plt.scatter(idx_n,y1, label = 'Kinetic luminosity')
#plt.scatter(x,y2,label = r'Kinetic energy derieved from $N_{tot}$')

plt.legend(frameon=False)
plt.hlines(mean_k,0,1+np.max(idx_n),label='Average luminosity',colors='red',linestyles='--')
plt.hlines(L_best_k,0,1+np.max(idx_n),label='Best-fit',colors='C2',linestyles='--')
'''
plt.hlines(mean_kn,0,1+np.max(idx_n),label='Average luminosity',colors='blue',linestyles='--')
plt.hlines(L_best_kn,0,1+np.max(idx_n),label='Best-fit',colors='C1',linestyles='--')
'''
plt.legend(frameon=False)
plt.legend()
plt.savefig('/home/wsy/V4641_Sgr/Luminosity_%spc.png'%R_pc)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

plt.figure()
plt.title('Minimum Kinetic Luminosity')
#plt.semilogx()
plt.semilogy()
plt.xlim(0,1+np.max(idx_n))
#plt.ylim(1e35,5e39)
plt.ylim(1e34,5e39)
plt.xlabel('Index')
plt.ylabel('Luminosity(erg/s)')
plt.scatter(idx_n,y2, label = 'Minimum Kinetic luminosity')
#plt.scatter(x,y2,label = r'Kinetic energy derieved from $N_{tot}$')

plt.legend(frameon=False)
plt.hlines(mean_kn,0,1+np.max(idx_n),label='Average luminosity',colors='red',linestyles='--')
plt.hlines(L_best_kn,0,1+np.max(idx_n),label='Best-fit',colors='C2',linestyles='--')
'''
plt.hlines(mean_kn,0,1+np.max(idx_n),label='Average luminosity',colors='blue',linestyles='--')
plt.hlines(L_best_kn,0,1+np.max(idx_n),label='Best-fit',colors='C1',linestyles='--')
'''
plt.legend(frameon=False)
plt.legend()
plt.savefig('/home/wsy/V4641_Sgr/Luminosity_%spc_minimum.png'%R_pc)
plt.show()


#-------------------------------------------------------------------------------------------------------------------------------
#表面亮度
# x,y均为无量纲量
x = spectrum_energy_syn
y = NAIMA_SYN(ene_best,n_best,lgB_best,beta_best,lgN_best)
y_eV = ((y*(u.erg)).to(u.eV)).value
n_eV = y_eV/x

#峰值流量和位置
mask_ymax = np.argmax(y)
print('The peak energy(eV): %s'%x[mask_ymax])
print('The peak flux(erg/s):%s'%y[mask_ymax])

#总流量
mask1_xrange =(x>=2e3) 
mask2_xrange = (x<=1e4)
mask_xrange = (mask1_xrange & mask2_xrange)
x_eV_range = x[mask_xrange]
y_eV_range = n_eV[mask_xrange]
F_tot = (((simps(y_eV_range,x_eV_range))*(u.eV)).to(u.erg)).value
print('Total flux of 2-10 keV(erg/s):%s'%F_tot)

#表面亮度 R_jet = ((5*(u.pc)).cgs).value, D_jet = ((100*(u.pc)).cgs).value
d = ((6.2*(u.kpc)).to(u.pc)).value
arcmin_2 = (2*R_pc/d)*3437.7467707849396**2*(100/d)
S_B = F_tot/arcmin_2
print('Surface Brightness(arcmin^-2):%s'%S_B)
