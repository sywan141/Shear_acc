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
    mask = np.all((flat_samples >= quantiles_low) & (flat_samples <= quantiles_high),axis=1)
    paras_subset_1sigma = flat_samples[mask]
    chi2_subset_1sigma = chi2_samples[mask]
    mask1=np.argmin(chi2_subset_1sigma)
    chi2_min = np.min(chi2_subset_1sigma)
    theta_best=paras_subset_1sigma[mask1]
    #print(chi2_min)

    delt_chi2 = chi2_subset_1sigma-chi2_min
    delt_crit = chi2.ppf(0.68, DOF)
    mask2 = (delt_chi2<=delt_crit)
    paras_subset_final = paras_subset_1sigma[mask2]
    length = len(paras_subset_final)
    mask_cut = np.linspace(0,length-1,num=100).astype(int)

    # record syn radiation
    lgB_best,rate_best,beta_best,lgN_best,lgrho_best = theta_best
    R_jet = ((i*(u.pc)).cgs).value
    spectrum_energy_syn = np.logspace(1,9,500)
    SYN_all = []
    IC_all = []
    with Pool(processes=100) as pool:
        results = pool.map(partial(process_paras, R=i), paras_subset_final[mask_cut]) 
    SYN_all = [r[0] for r in results]
    IC_all = [r[1] for r in results]
    ene_best,n_best=Elect_spec(i,lgB_best,rate_best,beta_best,lgN_best,lgrho_best)
    Flux_best = NAIMA_SYN(ene_best,n_best,lgB_best,beta_best,lgN_best)
    sed_min = np.min(SYN_all,axis=0)
    sed_max = np.max(SYN_all,axis=0)
    filei = np.column_stack((spectrum_energy_syn,Flux_best,sed_min,sed_max))
    #np.savetxt(tar_dir+'SYN_%spc_spec.txt'%i,filei)

    # calculate the integrated flux
    mask_ene1 = (2e3<=spectrum_energy_syn) & (spectrum_energy_syn<=1e4) # 2-10 keV
    #mask_ene2 = (2e3<=spectrum_energy_syn) & (spectrum_energy_syn<=7e3) # 2-7 keV
    x_range1 = spectrum_energy_syn[mask_ene1]
    y_range1 = ((Flux_best[mask_ene1]*(u.erg)).to(u.eV)).value/x_range1
    F_t = (((simps(y_range1,x_range1))*(u.eV)).to(u.erg)).value
    print(F_t)
    np.savetxt(tar_dir+'SYN_%spc_spec.txt'%i,filei)


'''
    F1 = []
    F2 = []
    for j in np.arange(len(SYN_all)):
        x = spectrum_energy_syn
        y = SYN_all[j]
        y_eV = ((y*(u.erg)).to(u.eV)).value
        n_eV = y_eV/x
        x_eV_range1 = x[mask_ene1]
        y_eV_range1 = n_eV[mask_ene1]
        F_tot1 = (((simps(y_eV_range1,x_eV_range1))*(u.eV)).to(u.erg)).value
        F1.append(F_tot1)
        x_eV_range2 = x[mask_ene2]
        y_eV_range2 = n_eV[mask_ene2]
        F_tot2 = (((simps(y_eV_range2,x_eV_range2))*(u.eV)).to(u.erg)).value
        F2.append(F_tot2)
    filei = np.column_stack((F1,F2))
    np.savetxt(tar_dir+'SYN_%spc_intflux.txt'%i,filei)'
'''