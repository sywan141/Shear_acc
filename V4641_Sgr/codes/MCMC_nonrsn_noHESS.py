# Condition with modified diffusion timescale, no resonance limit
import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from naima.models import TableModel
from naima.models import InverseCompton
from NAIMA_RAD import Elect_spec, NAIMA_IC, NAIMA_SYN
from scipy.integrate import simps
from scipy.stats import gaussian_kde,chi2
from multiprocessing import Pool,cpu_count
from functools import partial
import math
import emcee
import warnings
import corner
import datetime
warnings.filterwarnings("ignore")
import os
import random
import shutil
import sys

R = float(sys.argv[1])
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
factors = np.loadtxt('/home/wsy/V4641_Sgr/Extinction_factors.txt')[[5] + list(range(7, 17))]
q=5/3
T_cmb = 2.72 # CMB temperature
z=0
xi=1
d_pc = ((6.2*(u.kpc)).to(u.pc)).value
start = datetime.datetime.now()
dir_p='/home/wsy/V4641_Sgr/Rads/'
dir_t='/home/wsy/V4641_Sgr/Paras_1sigma/'
dir_data1 = '/home/wsy/V4641_Sgr/SYN_noHESS/'
dir_data2 = '/home/wsy/V4641_Sgr/IC_noHESS/'
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
'''
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
'''
a_all = np.row_stack((a1,a2))
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
    U_B = np.square(B0)/(8*np.pi)
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

    # calculate
    EC = TableModel(ene_eV,n_eV)
    IC = InverseCompton(EC,['CMB','NIR','FIR'],Eemax = 1e20*(u.eV))        # Threen types of photon fields
    spectrum_energy_ic = ene_obs*(u.TeV) # observed energies
    sed_IC = IC.sed(spectrum_energy_ic,distance=6.2*(u.kpc)) # erg/s
    sed_IC = N_tot*(sed_IC.value)
    rad = (72/360)*2*pi                                                    # The angle between the jet axis and the line of sight
    D = 1/(Gamma**4*(1-beta*math.cos(rad))**3)
    sed_IC = factors*D*sed_IC                                                      # Take the jet beaming effect into consideration(along with galactic extinction)
   
    return sed_IC
#-------------------------------------------------------------------------------------------------------

# The likely hood function is defined either by chi2 or gaussian probability function
def log_likelihood(theta,flux_obs,yerr):
    lgB,rate,beta,lgN,lgrho=theta
    sigma2 = (yerr)**2
    model = SSEPL(lgB,rate,beta,lgN,lgrho)
    if not np.isfinite(model.any()):
        return -np.inf
    #w = [0.9,0.9,0.9,0.9,0.9,0.8,0.9,0.8,0.8,1.1,1.1,0.8,1.1,0.8,1.1,1.1,0.5]
    chi2 = -0.5*np.sum((flux_obs-model)**2/sigma2) #+ np.log(sigma2)
    return chi2
    
def log_prior(theta):
    lgB,rate,beta,lgN,lgrho=theta
    lgB_min=-2
    B_min = (((10**lgB_min)*(u.uG)).to(u.G)).value
    lgB_max = np.log10((((2*c*np.sqrt(pi*10**lgrho))*(u.G)).to(u.uG)).value)
    lgrho_min = np.log10((B_min**2)/(4*pi*c**2))
    L_edd = 1e39  #Eddington luminosity(erg/s)
    Gamma = 1/np.sqrt(1-beta**2)
    rho_max = L_edd/(pi*R_jet**2*(Gamma-1)*beta*c**3)
    lgrho_max = np.log10(rho_max)
    lgN_max = np.log10(((10**lgrho)/m_p)*pi*R_jet**2*D_jet)
    if  (lgB_min<lgB<lgB_max) and (0<rate<1) and (0.3<beta<1.0) and (30.0<lgN<lgN_max) and (lgrho_min<lgrho<lgrho_max):
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

def process_paras(paras, R):
    lgB, rate, beta, lgN, lgrho = paras
    ene_eV, n_eV = Elect_spec(R, lgB, rate, beta, lgN, lgrho)
    sed_SYN = NAIMA_SYN(ene_eV, n_eV, lgB, beta, lgN)
    sed_IC = NAIMA_IC(ene_eV, n_eV, beta, lgN)[0]
    return sed_SYN, sed_IC
#-------------------------------------------------------------------------------------------------------
num = 1
nwalkers = 64
ndim = 5 
n_points = 11
DOF = n_points-ndim
initial = 10000
N = 50000
initial_steps = initial
n_steps = N

[lgB0,rate0,beta0,lgN0,lgrho0] = [-0.12136061649510149, 0.6358006576545763, 0.6720728553300879, 44.828848441410436, -32.4]
initial = [lgB0,rate0,beta0,lgN0,lgrho0]

random.seed(num)
p0 = initial + np.random.randn(nwalkers, ndim)*[0.01,0.01,0.01,0.01,0.01]
optimal_processes = 100

if __name__ == '__main__':

    with Pool(processes = optimal_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(flux_obs, err_obs),pool=pool,moves=[(emcee.moves.StretchMove(), 0.05)])#,moves=[(emcee.moves.StretchMove(), 0.5)]
        state = sampler.run_mcmc(p0, initial_steps, progress=True)
        sampler.reset()
        sampler.run_mcmc(state,n_steps,progress=True)
        tau =sampler.get_autocorr_time(tol=0)
#-------------------------------------------------------------------------------------------------------
    burnin =int(8*np.max(tau))
    thin = int(0.1*np.min(tau))
    
    # Gelman-Rubin coef
    samples = sampler.get_chain(discard = burnin,thin=thin)
    flat_samples = sampler.get_chain(discard = burnin,thin=thin,flat=True)
    fig, axes = plt.subplots(ndim,figsize=(10, 7), sharex=True)

#-------------------------------------------------------------------------------------------------------
    new_folder = "A_nonrsn_MFP_noHESS_xi=1_%spc_limited_DIFFUSION_MODIFIED_%s_GalacticExtinction_q=%s_widerange"%(R,num,q)

    path1 ='/home/wsy/V4641_Sgr/results/paras'
    target_dir1 = os.path.join(path1, new_folder)
    if os.path.exists(target_dir1):
        shutil.rmtree(target_dir1)
    os.chdir(path1)
    os.makedirs(new_folder)
    labels = ['lgB', r'$\eta$',r'$\beta_0$',r'$lgN_{tot}$',r'$lg\rho$']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step number")
    plt.savefig(path1+'/'+new_folder+'/lgB0=%s_rate=%s_beta=%s_lgN=%s_lgrho=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],initial[3],initial[4],n_steps))
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

    proba_list = sampler.get_log_prob(discard = burnin,thin=thin,flat=True) # All log probability, lp+lk, flat
    paras_list = sampler.get_chain(discard = burnin,thin=thin,flat=True) # All parameter lists, flat
    np.savetxt(path1+'/'+new_folder+'/chi2_lgB0=%s_rate=%s_beta=%s_lgN=%s_lgrho=%s_n=%s_weighted.txt'%(initial[0],initial[1],initial[2],initial[3],initial[4],n_steps),-2*proba_list)
    np.savetxt(path1+'/'+new_folder+'/paras_lgB0=%s_rate=%s_beta=%s_lgN=%s_lgrho=%s_n=%s_weighted.txt'%(initial[0],initial[1],initial[2],initial[3],initial[4],n_steps),paras_list)

    quantiles_low = np.percentile(flat_samples, 16, axis=0)  # 0.16
    quantiles_high = np.percentile(flat_samples, 84, axis=0) # 0.84

# select samples(subsection), which are in the 1-sigma confidence range of all 5 paras
    mask = np.all((flat_samples >= quantiles_low) & (flat_samples <= quantiles_high),axis=1)
    fsample_subset_1sigma = flat_samples[mask]
    fproba_subset_1sigma = proba_list[mask]
    paras_subset_1sigma = paras_list[mask]

    mask1= np.argmax(fproba_subset_1sigma) # maximum probability in the 1-sigma confidence range (minimum chi2)
    value = paras_subset_1sigma[mask1]
    print(-2*np.max(fproba_subset_1sigma))
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

    fig = corner.corner(flat_samples, 
                        labels=labels,
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                        quantiles=(0.16,0.5,0.84),
                        hist_kwargs={"density": True}) # quantiles: 1-sigma range(68%)
    
    # Locate the highest probability with KDE
    peaks = []
    for i in range(ndim):
        data = flat_samples[:, i]
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 1000)
        kde_vals = kde(x)
        peak_x = x[np.argmax(kde_vals)]
        peaks.append(peak_x) # parametre sets for the maximum MAP

    # get axes from the corner plot
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Plot the KDE curves in the diagonal subplots 
    for i in range(ndim):
        ax = axes[i, i]
        data = flat_samples[:, i]
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 1000)
        y = kde(x)
        ax.plot(x, y, color='deepskyblue', lw=1.5, ls='--', label='KDE')

    corner.overplot_lines(fig, value, color="C1") # parametres when chi2 reaches minimum
    corner.overplot_points(fig, value[None], marker="s", color="C1")

    plt.legend()
    plt.savefig(path2+'/'+new_folder+'/lgB0=%s_rate=%s_beta=%s_lgN=%s_lgrho=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],initial[3],initial[4],n_steps))
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
    plt.savefig(path1+'/'+new_folder+'/chi2plot_lgB0=%s_rate=%s_beta=%s_lgN=%s_lgrho=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],initial[3],initial[4],n_steps))

    #------------------------------------------------------------------------------------------
    # 1-sigma range
    chi2_subset_1sigma =-2*fproba_subset_1sigma
    chi2_min = chi2_subset_1sigma[mask1]
    delt_chi2 = chi2_subset_1sigma-chi2_min
    delt_crit = chi2.ppf(0.68, DOF)
    mask2 = (delt_chi2<=delt_crit)
    paras_subset_final = paras_subset_1sigma[mask2]
    length = len(paras_subset_final)
    mask_cut = np.linspace(0,length-1,num=100).astype(int)


    # plot syn radiation
    lgB_best,rate_best,beta_best,lgN_best,lgrho_best = value
    spectrum_energy_syn = np.logspace(1,8,500)
    SYN_all = []
    IC_all = []
    with Pool(processes=optimal_processes) as pool:
        results = pool.map(partial(process_paras, R=R), paras_subset_final[mask_cut]) 
    SYN_all = [r[0] for r in results]
    IC_all = [r[1] for r in results]
    plt.figure()
    plt.title('V4641 Sgr')
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e2,1e6)
    if (R<= 0.1):
        plt.ylim(1e-13,1e-9)
    else:
        plt.ylim(1e-15,1e-11)
    plt.xlabel('Energy(eV)')
    plt.ylabel(r'Flux(erg $cm^{-2}s^{-1}$)')
    SYN_all=np.array(SYN_all)
    sed_min = np.min(SYN_all,axis=0)
    sed_max = np.max(SYN_all,axis=0)
    plt.fill_between(spectrum_energy_syn, sed_min, sed_max, color="lightcoral", label=r" SEDs in 1$ \sigma$, $R_{jet} = %s pc$"%R)
    ene_best,n_best=Elect_spec(R,lgB_best,rate_best,beta_best,lgN_best,lgrho_best)
    S_bxrism = 7*10**(-15)
    arcmin_2 = (2*R/(d_pc))*3437.7467707849396**2*(100/d_pc)
    Flux_max = S_bxrism*arcmin_2
    Flux_best = NAIMA_SYN(ene_best,n_best,lgB_best,beta_best,lgN_best)
    plt.plot(spectrum_energy_syn,Flux_best,c='red',label='Best fit values')
    plt.hlines(Flux_max,1e2,1e6,colors='deepskyblue',linestyle = '-.',label=r'Intergrated flux from XRISM($S_B \Delta \Omega$)')
    plt.legend(frameon=False)
    plt.savefig(dir_p + 'SYN_%spc_noHESS.png'%R)
    plt.show()

    # save txt files
    file1 = np.column_stack((spectrum_energy_syn,Flux_best,sed_min,sed_max))
    np.savetxt(dir_data1+'SYN_%spc_spec.txt'%R,file1)
    

    # plot IC radiation
    spectrum_energy_ic = np.logspace(1,4,2000)
    plt.figure()
    plt.title('V4641 Sgr')
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e1,1e4)
    plt.ylim(3e-15,1e-10)
    plt.xlabel('Energy(TeV)')
    plt.ylabel(r'Flux(erg $cm^{-2}s^{-1}$)')
    IC_all=np.array(IC_all)
    sed_min = np.min(IC_all,axis=0)
    sed_max = np.max(IC_all,axis=0) # minimum in each row
    plt.fill_between(spectrum_energy_ic, sed_min, sed_max, color="lightskyblue", label=r" SEDs in 1$ \sigma$, $R_{jet} = %s pc $"%R)
    #ene_best,n_best=Elect_spec(lgB_best,rate_best,beta_best,lgN_best,lgrho_best)
    IC_tot, IC_cmb, IC_fir, IC_nir, IC_obs= NAIMA_IC(ene_best,n_best,beta_best,lgN_best)
    plt.plot(spectrum_energy_ic,IC_tot,c='blue',label='Best fit values(Total)')
    plt.plot(spectrum_energy_ic,IC_cmb,c='C1',linestyle='--', label='Best fit values(CMB)')
    plt.plot(spectrum_energy_ic,IC_fir,c='C2',linestyle='--', label='Best fit values(FIR)')
    plt.plot(spectrum_energy_ic,IC_nir,c='cyan',linestyle='--',label='Best fit values(NIR)')
    plt.errorbar(x1,y1,c='darkviolet',fmt = 'o', yerr = error_limit1, ls='None',label = 'HAWC')
    plt.errorbar(x2,y2,c='firebrick',fmt = 'D', yerr = error_limit2, ls='None',label = 'LHAASO')
    #plt.errorbar(x3,y3,c='forestgreen',fmt = 's', yerr = error_limit3 ,ls='None',label = 'HESS')
    plt.legend(frameon=False)
    plt.savefig(dir_p + 'IC_%spc_noHESS.png'%R)
    plt.show()

    # save txt files
    file2 = np.column_stack((spectrum_energy_ic,IC_tot, IC_cmb, IC_fir, IC_nir, sed_min, sed_max))
    np.savetxt(dir_data2+'IC_%spc_spec.txt'%R,file2)

    # print 1-sigma limit for each parametre
    rows = []
    for i in range(ndim):
        param_col = paras_subset_final[:, i]
        param_min = param_col.min()
        param_max = param_col.max()
        val = value[i]
        rows.append([param_min, val, param_max, val - param_min, param_max - val])
    np.savetxt(dir_t + 'Range_%spc_noHESS.txt' % R, rows)
    end = datetime.datetime.now()
    print('Total time:%s'%(end-start))