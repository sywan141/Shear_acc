# Condition with modified diffusion timescale
import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from naima.models import TableModel
from naima.models import InverseCompton
from scipy.integrate import simps
from multiprocessing import Pool,cpu_count
import math
import emcee
import warnings
import corner
warnings.filterwarnings("ignore")
import os
import random
import shutil

R = 5.0
m_e = (const.m_e.cgs).value
c= (const.c.cgs).value
e = (const.e.esu).value
sigma_sb = (const.sigma_sb.cgs).value
sigma_e = (8*np.pi/3)*np.square(e**2/(m_e*c**2))
m_p = (const.m_p.cgs).value
R_jet = ((R*(u.pc)).cgs).value
D_jet = ((100*(u.pc)).cgs).value
T_cmb = 2.72 # CMB temperature
pi = np.pi
inf = np.inf
nan = float('nan')
#E_pm = ((0.8*(u.PeV)).to(u.erg)).value
#gamma_pm=E_pm/(m_e*c**2)
factors = np.loadtxt('/home/wsy/V4641_Sgr/Extinction_factors.txt')
q=5/3
z=0
xi=1
q_rsn = 0

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
err1 = (upper1+lower1)/2 # average
a1 = np.column_stack((x1,y1,err1))
#LHAASO,in TeV
x2 = (file2[2:,0]*(u.TeV)).value
y2 = ((file2[2:,1]*(u.TeV)).to(u.erg)).value
upper2 = ((file2[2:,2]*(u.TeV)).to(u.erg)).value
lower2 = ((file2[2:,3]*(u.TeV)).to(u.erg)).value
err2 = (upper2+lower2)/2
a2 = np.column_stack((x2,y2,err2))
# HESS, in erg
x3 = (file3[:,0]*(u.TeV)).value
y3 = ((file3[:,1]*(u.erg))).value
upper3 = (file3[:,2]*(u.erg)).value-y3
lower3 = y3-(file3[:,3]*(u.erg)).value
err3 = (upper3+lower3)/2
a3 = np.column_stack((x3,y3,err3))
a_all = np.row_stack((a1,a2,a3))
sorted_a_all = a_all[np.argsort(a_all[:, 0])]

ene_obs = np.float64(sorted_a_all[:,0])
flux_obs = np.float64(sorted_a_all[:,1])
err_obs = np.float64(sorted_a_all[:,2])
if np.any(np.isnan(flux_obs)) or np.any(np.isnan(1/err_obs)):
    raise ValueError("观测数据包含 NaN 值")
#-------------------------------------------------------------------------------------------------------
# Combined spectrum model of stochastic and shear acceleration
def coeff_A1(D_sh,xi,Lam_max,q,B0):
    return D_sh*xi**(-1)*(Lam_max/c)**(q-1)*(m_e*c/(e*B0))**(2-q)

def coeff_A2(B0):
    U_B = np.square(B0)/8*np.pi
    U_cmb = ((4*sigma_sb)/c)*T_cmb**4*(1+z)**4     
    U_rad = U_cmb
    return (sigma_e*B0**2*(1+U_rad/U_B))/(6*np.pi*m_e*c)

# averaged
def coeff_A3(D_sh,A1):
    return (9*R_jet**2*D_sh)/(2*c**2*A1)

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

def SSEPL(lgB,rate,beta,lgrho,lgK1,lgK2):
    K1=10**lgK1
    K2=10**lgK2
#------------------------------------------------------------------------------------------------------------------
    # free and fixed parameters
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
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
    
    # Hillas
    gamma_cut =  e*B0/(m_e*c**2)*np.sqrt(xi*Lam_max*R_jet)  # MFP when timescale is changed
    #gamma_cut = e*B0*R_jet/(m_e*c**2)                                 

    # spectrum after gamma_rsn
    gamma_gyr=np.logspace(np.log10(gamma_rsn),np.log10(gamma_cut),1000)
    b_s = 1-q_rsn
    w = 10*rate**2/(beta**2*Gamma_j4) # radius escaping term
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

    C2 = K2
    C1 =  (n_gyr[0]-C2*F2(w,q,gamma_rsn,z_cut_1))/F1(w,q,gamma_rsn,z_cut_1)
    n_1 = C1*F1(w,q,gamma_1,z_1)+C2*F2(w,q,gamma_1,z_1)
    if(n_1[-1]>=n_1[-2]):
        return np.zeros(1)-np.inf
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

    # calculate
    EC = TableModel(ene_eV,n_eV)
    #sed_SYN = SYN.sed(spectrum_energy_syn, distance=6.2 * u.kpc)
    IC = InverseCompton(EC,['CMB','NIR','FIR'],Eemax = 1e20*(u.eV))        # Threen types of photon fields
    spectrum_energy_ic = ene_obs*(u.TeV) # observed energies
    sed_IC = IC.sed(spectrum_energy_ic,distance=6.2*(u.kpc)) # erg/s
    #sed_IC=N_tot*(sed_IC.value)
    rad = (72/360)*2*pi                                                    # The angle between the jet axis and the line of sight
    D = 1/(Gamma**4*(1-beta*math.cos(rad))**3)
    sed_IC = factors*D*(sed_IC.value)
    
    return sed_IC
#-------------------------------------------------------------------------------------------------------

# The likely hood function is defined either by chi2 or gaussian probability function
def log_likelihood(theta,flux_obs,yerr):
    lgB,rate,beta,lgrho,lgK1,lgK2=theta
    sigma2 = (yerr)**2
    model = SSEPL(lgB,rate,beta,lgrho,lgK1,lgK2)
    if not np.isfinite(model.any()):
        return -np.inf
    #w = [1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,1]
    chi2 = -0.5*np.sum((flux_obs-model)**2/sigma2) #+ np.log(sigma2)
    return chi2
    
def log_prior(theta):
    lgB,rate,beta,lgrho,lgK1,lgK2=theta
    Reg = rate*R_jet
    #Lam_max = Reg
    lgB_min=-2
    B_min = (((10**lgB_min)*(u.uG)).to(u.G)).value
    lgB_max = np.log10((((2*c*np.sqrt(pi*10**lgrho))*(u.G)).to(u.uG)).value)
    lgrho_min = np.log10((B_min**2)/(4*pi*c**2))
    L_edd = 1e39  #Eddington luminosity(erg/s)
    Gamma = 1/np.sqrt(1-beta**2)
    rho_max = L_edd/(pi*R_jet**2*(Gamma-1)*beta*c**3)
    lgrho_max = np.log10(rho_max)
    #lgN_max = np.log10(((10**lgrho)/m_p)*pi*R_jet**2*D_jet)
    if  (lgB_min<lgB<lgB_max) and (0<rate<1) and (0.0<beta<1.0) and (lgrho_min<lgrho<lgrho_max) and (lgK1>64) and (lgK2<lgK1):
        return 0.0
    return -np.inf

def log_probability(theta, flux_obs, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf # impossible prior values
    lk = log_likelihood(theta, flux_obs, yerr)
    if (math.isnan(lk)):
        return -np.inf
    return lp + lk
#-------------------------------------------------------------------------------------------------------
num = 0
nwalkers = 64
ndim = 6
initial = 10000
N = 50000
initial_steps = initial
n_steps = N

[lgB0,rate0, beta0,lgrho0,lgK10,lgK20] = [ -0.30377436  ,0.21362  ,0.6406463 ,-30.97161229  ,64.59506544, 51.34049404]
initial = [lgB0,rate0,beta0,lgrho0,lgK10,lgK20]

random.seed(num)
p0 = initial + np.random.randn(nwalkers, ndim)*(1e-3)
optimal_processes = cpu_count()

if __name__ == '__main__':

    with Pool(processes = optimal_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(flux_obs, err_obs),pool=pool,moves=[(emcee.moves.StretchMove(), 0.05)])#,moves=[(emcee.moves.StretchMove(), 0.5)]
        state = sampler.run_mcmc(p0, initial_steps, progress=True)
        sampler.reset()
        sampler.run_mcmc(state,n_steps,progress=True)
        tau =sampler.get_autocorr_time(tol=0)
#-------------------------------------------------------------------------------------------------------
    #burnin =int(8*np.max(tau))
    #thin = int(0.2*np.min(tau))
    burnin =int(8*np.max(tau))
    thin = int(0.1*np.min(tau))
    #burnin=1
    #thin=1
    # Gelman-Rubin coef
    samples = sampler.get_chain(discard = burnin,thin=thin)
    flat_samples = sampler.get_chain(discard = burnin,thin=thin,flat=True)
    fig, axes = plt.subplots(ndim,figsize=(10, 7), sharex=True)

#-------------------------------------------------------------------------------------------------------
    new_folder = "Averaged_MFP_Cooling_free_K1K2_FP_GYR_NofixedLam_xi=1_%spc_limited_DIFFUSION_MODIFIED_%s_GalacticExtinction_q=%s_widerange"%(R,num,q)
    #new_folder = "All_paras_smallLam_chi2_fixedRsh_1pc_0.5pc_01"
    #new_folder = "All_paras_smallLam_chi2_fixedRsh_0.5pc_0.25pc_02"
    #new_folder = "All_paras_smallLam_chi2_fixedRsh_2.5pc_2.5pc_promptcut_01"

    path1 ='/home/wsy/V4641_Sgr/results/paras'
    target_dir1 = os.path.join(path1, new_folder)
    if os.path.exists(target_dir1):
        shutil.rmtree(target_dir1)

    os.chdir(path1)
    os.makedirs(new_folder)
    labels = ['lgB', r'$\eta$',r'$\beta_0$',r'$lg\rho$',r'$lgK_1$',r'$lgK_2$']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step number")
    plt.savefig(path1+'/'+new_folder+'/lgB0=%s_rate=%s_beta=%s_lgrho=%s_lgK1=%s_lgK2=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],n_steps))
    plt.show()
#-------------------------------------------------------------------------------------------------------
    path2 = '/home/wsy/V4641_Sgr/results/corners'
    target_dir2 = os.path.join(path2, new_folder)
    if os.path.exists(target_dir2):
        shutil.rmtree(target_dir2)
    
    os.chdir(path2)
    os.makedirs(new_folder)

# fitting results(between 1-sigma:16%,50%,84%)
# median values
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = r"$\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}$"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        print(txt)

    chi2_list = sampler.get_log_prob(flat=True) # nwalkers*ndim
    paras_list = sampler.get_chain(flat=True)
    np.savetxt(path1+'/'+new_folder+'/chi2_lgB0=%s_rate=%s_beta=%s_lgrho=%s_lgK1=%s_lgK2=%s_n=%s_weighted.txt'%(initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],n_steps),chi2_list)
    np.savetxt(path1+'/'+new_folder+'/paras_lgB0=%s_rate=%s_beta=%s_lgrho=%s_lgK1=%s_lgK2=%s_n=%s_weighted.txt'%(initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],n_steps),paras_list)

    mask1= np.argmax(chi2_list)
    print(paras_list[mask1])
    print(-2*np.max(chi2_list))
#-------------------------------------------------------------------------------------------------------

# save models
    path3 ='/home/wsy/V4641_Sgr/results/models'
    target_dir3 = os.path.join(path3, new_folder)
    if os.path.exists(target_dir3):
        shutil.rmtree(target_dir3)

    os.chdir(path3)
    os.makedirs(new_folder)
    np.save(path3+'/'+new_folder+"flat_samples.npy", flat_samples)

#-------------------------------------------------------------------------------------------------------

    # corner plot
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step number")
    plt.show()
    #flat_samples = sampler.get_chain(discard = int(0.5*n_steps),thin=30, flat=True)
    #print(flat_samples.shape)
    fig = corner.corner(flat_samples, labels=labels,show_titles=True,title_kwargs={"fontsize": 12},quantiles=(0.16,0.5,0.84))
    #value1 = np.mean(flat_samples,axis=0)
    value2 = paras_list[mask1]
    #corner.overplot_lines(fig, value1, color="C1")
    #corner.overplot_points(fig, value1[None], marker="s", color="C1")
    corner.overplot_lines(fig, value2, color="C1")
    corner.overplot_points(fig, value2[None], marker="s", color="C1")
    plt.savefig(path2+'/'+new_folder+'/lgB0=%s_rate=%s_beta=%s_lgrho=%s_lgK1=%s_lgK2=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],n_steps))
    plt.show()


#-------------------------------------------------------------------------------------------------------
# plot chi2
    log_probabilities = sampler.get_log_prob()  # shape: (n_steps, n_walkers)
    iters = np.arange(log_probabilities.shape[0])
    plt.figure(figsize=(10, 6))
    for walker in range(log_probabilities.shape[1]):  # 遍历每个 walker
        plt.plot(iters, -2 * log_probabilities[:, walker], alpha=0.5, label=f'Chain {walker+1}') # start from 0
        plt.xlabel('Iteration')
        plt.ylabel('Chi-squared')
        plt.title('Chi-squared for Each Chain')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
    plt.savefig(path1+'/'+new_folder+'/chi2plot_lgB0=%s_rate=%s_beta=%s_lgrho=%s_lgK1=%s_lgK2=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],n_steps))