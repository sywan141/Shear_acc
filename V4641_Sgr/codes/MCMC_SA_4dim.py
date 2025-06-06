# Condition with modified diffusion timescale, no resonance limit
import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import hyp1f1
import matplotlib.pyplot as plt
from naima.models import TableModel
from naima.models import InverseCompton
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
factors = np.loadtxt('/home/wsy/V4641_Sgr/Extinction_factors.txt')
q=2
T_cmb = 2.72 # CMB temperature
z=0
xi=1
d_pc = ((6.2*(u.kpc)).to(u.pc)).value
start = datetime.datetime.now()
#dir_p='/home/wsy/V4641_Sgr/Rads/'
#dir_t='/home/wsy/V4641_Sgr/Paras_1sigma/'
#dir_data1 = '/home/wsy/V4641_Sgr/SYN_HESS/'
#dir_data2 = '/home/wsy/V4641_Sgr/IC_HESS/'

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
if np.any(np.isnan(flux_obs)) or np.any(np.isnan(1/err_obs)):
    raise ValueError("观测数据包含 NaN 值")
#-------------------------------------------------------------------------------------------------------
def IC_RAD(lgB,rate,lgK,lgrho):
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
    #Lam_max = rate*R_jet
    K_all = 10**lgK # Normalization index, 10**Norm_A
    #lgrho = -32
    rho = 10**lgrho
    lgE_min = 6
    gamma_min = ((((10**lgE_min)*(u.eV)).to(u.erg))/(m_e*c**2)).value
    lg_min = np.log10(gamma_min)
    # calculate the coefficients for stochastic acc
    beta_Alf = B0/(c*np.sqrt(4*pi*rho)) # Alfven velocity +B0**2/(4*pi)
    Gamma_Alf = 1/np.sqrt(1-beta_Alf**2)
    Lam_max=rate*R_jet
    U_B = np.square(B0)/(8*np.pi)
    U_cmb = ((4*sigma_sb)/c)*T_cmb**4*(1+z)**4 # only CMB is considered
    U_rad = U_cmb
    A1 = ((xi*Gamma_Alf**4*beta_Alf**2*c)/(Lam_max)**(q-1))*(m_e*c**2/(e*B0))**(q-2)
    A2 = (sigma_e*B0**2*(1+U_rad/U_B))/(6*np.pi*m_e*c)
    A3 = ((3*R_jet**2*xi*Lam_max**(1-q))/(2*c))*((e*B0)/(m_e*c**2))**(2-q)
    
    # This part is for the shearing spectrum
    gamma_max1 = ((2*A2)/((2+q)*A1))**(1/(q-3)) # cooling limit
    gamma_max2 = e*B0*R_jet/(m_e*c**2) # hillas condition
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
    
    length = len(n)
    if (length<=5):
        return np.zeros(1)-np.inf

    ene_eV = ((gamma_all*m_e*c**2)*(u.erg)).to(u.eV) # energy in eV
    n_eV = n/((m_e*c**2)*(u.erg)).to(u.eV) # number density(1/eV)
    
    EC = TableModel(ene_eV,n_eV)
    IC = InverseCompton(EC,['CMB','FIR','NIR'],Eemax = 1e20*(u.eV))
    spectrum_energy_ic = ene_obs*(u.TeV) # observed energies
    sed_IC = (IC.sed(spectrum_energy_ic,distance=6.2*(u.kpc))).value # erg/s
    sed_IC = K_all*factors*sed_IC 
    
    return sed_IC
#-------------------------------------------------------------------------------------------------------

# The likely hood function is defined either by chi2 or gaussian probability function
def log_likelihood(theta,flux_obs,yerr):
    lgB,rate,lgK,lgrho=theta
    sigma2 = (yerr)**2
    model = IC_RAD(lgB,rate,lgK,lgrho)
    if not np.isfinite(model.any()):
        return -np.inf
    #w = [0.9,0.9,0.9,0.9,0.9,0.8,0.9,0.8,0.8,1.1,1.1,0.8,1.1,0.8,1.1,1.1,0.5]
    chi2 = -0.5*np.sum((flux_obs-model)**2/sigma2) #+ np.log(sigma2)
    return chi2
    
def log_prior(theta):
    lgB,rate,lgK,lgrho=theta
    #lgrho = -32
    lgB_min=-2
    B0=(((10**lgB)*(u.uG)).to(u.G)).value
    rho = 10**lgrho
    beta_Alf = (B0/np.sqrt(4*pi*((rho*c**2))))
    B_min = (((10**lgB_min)*(u.uG)).to(u.G)).value
    lgB_max = np.log10((((2*c*np.sqrt(pi*10**lgrho))*(u.G)).to(u.uG)).value)
    lgrho_min = np.log10((B_min**2)/(4*pi*c**2))
    #L_edd = 1e39  #Eddington luminosity(erg/s)
    #beta = 0.1
    #Gamma = 1/np.sqrt(1-beta**2)
    #rho_max = L_edd/(pi*R_jet**2*(Gamma-1)*beta*c**3)
    #lgrho_max = np.log10(rho_max)
    if  (lgB_min<lgB<lgB_max) and (0<rate<1): #and (lgrho_min<lgrho<lgrho_max) and (50<lgK<53):
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
num = 1
nwalkers = 64
ndim = 4
n_points = 17
DOF = n_points-ndim
initial = 1000
N = 100000
initial_steps = initial
n_steps = N

[lgB0,rate0,lgK0,lgrho0] = [-0.60, 0.040961672771081724 , 52.22, -32]
initial = [lgB0,rate0,lgK0,lgrho0]

random.seed(num)
p0 = initial + np.random.randn(nwalkers, ndim)*(1e-3)
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
    new_folder = "SA_4dim_model_xi=1_%spc_limited_DIFFUSION_MODIFIED_%s_GalacticExtinction_q=%s_widerange"%(R,num,q)

    path1 ='/home/wsy/V4641_Sgr/results/paras'
    target_dir1 = os.path.join(path1, new_folder)
    if os.path.exists(target_dir1):
        shutil.rmtree(target_dir1)
    os.chdir(path1)
    os.makedirs(new_folder)
    labels = ['lgB',r'$\eta$',r'$lgK$',r'$lg\rho$']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("step number")
    plt.savefig(path1+'/'+new_folder+'/lgB0=%s_rate=%s_lgK=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],n_steps))
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
    chi2_list = -2*proba_list
    np.savetxt(path1+'/'+new_folder+'/chi2_lgB0=%s_rate=%s_lgK=%s_n=%s_weighted.txt'%(initial[0],initial[1],initial[2],n_steps),chi2_list)
    np.savetxt(path1+'/'+new_folder+'/paras_lgB0=%s_rate=%s_lgK=%s_n=%s_weighted.txt'%(initial[0],initial[1],initial[2],n_steps),paras_list)


# select samples(subsection), which are in the 1-sigma confidence range of all 5 paras
# 在双峰/有偏分布的情况下，额外选取在一个sigma内的所有样本没有任何意义。最佳拟合值可能为两个峰中的任意一个

    mask1= np.argmin(chi2_list) # maximum probability in the 1-sigma confidence range (minimum chi2)
    value = paras_list[mask1]
    print(chi2_list[mask1])
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
    plt.savefig(path2+'/'+new_folder+'/lgB0=%s_rate=%s_lgK=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],n_steps))
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
    plt.savefig(path1+'/'+new_folder+'/chi2plot_lgB0=%s_rate=%s_lgK=%s_n=%s_weighted.png'%(initial[0],initial[1],initial[2],n_steps))