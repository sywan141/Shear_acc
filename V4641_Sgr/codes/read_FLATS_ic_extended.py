import numpy as np
from scipy.stats import chi2
import astropy.units as u
from multiprocessing import Pool
from NAIMA_RAD import Elect_spec, NAIMA_IC, NAIMA_SYN
from functools import partial
from scipy.integrate import simps

datadir = '/home/wsy/V4641_Sgr/results/paras/'
tar_dir = '/home/wsy/V4641_Sgr/read_flatsamples/'
DOF = 17-5
R=[5.0,1.0,0.5,0.1]

def process_paras(paras, R):
    lgB, rate, beta, lgN, lgrho = paras
    ene_eV, n_eV = Elect_spec(R, lgB, rate, beta, lgN, lgrho)
    sed_SYN = NAIMA_SYN(ene_eV, n_eV, lgB, beta, lgN)
    sed_IC = NAIMA_IC(ene_eV, n_eV, beta, lgN)[0]
    return sed_SYN, sed_IC

for i in R:
    flat_samples = np.loadtxt(datadir + 'A_nonrsn_MFP_xi=1_%spc_limited_DIFFUSION_MODIFIED_1_GalacticExtinction_q=1.6666666666666667_widerange/'%i+
    'paras_lgB0=-0.3_rate=0.2_beta=0.5851197_lgN=47.08667333_lgrho=-32.0421692_n=50000_weighted.txt')
    chi2_samples = np.loadtxt(datadir+'A_nonrsn_MFP_xi=1_%spc_limited_DIFFUSION_MODIFIED_1_GalacticExtinction_q=1.6666666666666667_widerange/'%i+
    'chi2_lgB0=-0.3_rate=0.2_beta=0.5851197_lgN=47.08667333_lgrho=-32.0421692_n=50000_weighted.txt')
    quantiles_low = np.percentile(flat_samples, 16, axis=0)  # 0.16
    quantiles_high = np.percentile(flat_samples, 84, axis=0) # 0.84
    mask = np.all((flat_samples >= quantiles_low) & (flat_samples <= quantiles_high),axis=1) # parameters in the 1 sigma range
    paras_subset_1sigma = flat_samples[mask]
    chi2_subset_1sigma = chi2_samples[mask]
    mask1=np.argmin(chi2_subset_1sigma)
    chi2_min = np.min(chi2_subset_1sigma)
    theta_best=paras_subset_1sigma[mask1]
    print(theta_best)

    delt_chi2 = chi2_subset_1sigma-chi2_min
    delt_crit = chi2.ppf(0.68, DOF)
    mask2 = (delt_chi2<=delt_crit)
    paras_subset_final = paras_subset_1sigma[mask2]
    length = len(paras_subset_final)
    mask_cut = np.linspace(0,length-1,num=100).astype(int)

    # record ic radiation
    lgB_best,rate_best,beta_best,lgN_best,lgrho_best = theta_best
    R_jet = ((i*(u.pc)).cgs).value
    spectrum_energy_ic = np.logspace(-3.1,3.2,1000)
    #SYN_all = []
    IC_all = []
    with Pool(processes=100) as pool:
        results = pool.map(partial(process_paras, R=i), paras_subset_final[mask_cut]) 
    #SYN_all = [r[0] for r in results]
    IC_all = [r[1] for r in results]
    ene_best,n_best=Elect_spec(i,lgB_best,rate_best,beta_best,lgN_best,lgrho_best)
    IC_tot, IC_cmb, IC_fir, IC_nir, IC_obs= NAIMA_IC(ene_best,n_best,beta_best,lgN_best)
    sed_min = np.min(IC_all,axis=0)
    sed_max = np.max(IC_all,axis=0)
    filei = np.column_stack((spectrum_energy_ic,IC_tot, IC_cmb, IC_fir, IC_nir, sed_min, sed_max))
    np.savetxt(tar_dir+'IC_%spc_spec_extended.txt'%i,filei)