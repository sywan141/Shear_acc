import numpy as np
from scipy.stats import chi2
import astropy.units as u
from multiprocessing import Pool
from NAIMA_RAD import Elect_spec, NAIMA_IC, NAIMA_SYN
from functools import partial

datadir = '/home/wsy/V4641_Sgr/results/paras/'
tar_dir = '/home/wsy/V4641_Sgr/codes/read_flatsamples/'
DOF = 11-5
R=[5.0,1.0,0.5,0.1]

def process_paras(paras, R):
    lgB, rate, beta, lgN, lgrho = paras
    ene_eV, n_eV = Elect_spec(R, lgB, rate, beta, lgN, lgrho)
    sed_SYN = NAIMA_SYN(ene_eV, n_eV, lgB, beta, lgN)
    sed_IC = NAIMA_IC(ene_eV, n_eV, beta, lgN)[0]
    return sed_SYN, sed_IC

for i in R:
    flat_samples = np.loadtxt(datadir + 'A_nonrsn_MFP_noHESS_xi=1_%spc_limited_DIFFUSION_MODIFIED_1_GalacticExtinction_q=1.6666666666666667_widerange/'%i+
    'paras_lgB0=-0.12136061649510149_rate=0.6358006576545763_beta=0.6720728553300879_lgN=44.828848441410436_lgrho=-32.4_n=50000_weighted.txt')
    chi2_samples = np.loadtxt(datadir+'A_nonrsn_MFP_noHESS_xi=1_%spc_limited_DIFFUSION_MODIFIED_1_GalacticExtinction_q=1.6666666666666667_widerange/'%i+
    'chi2_lgB0=-0.12136061649510149_rate=0.6358006576545763_beta=0.6720728553300879_lgN=44.828848441410436_lgrho=-32.4_n=50000_weighted.txt')
    quantiles_low = np.percentile(flat_samples, 16, axis=0)  # 0.16
    quantiles_high = np.percentile(flat_samples, 84, axis=0) # 0.84
    mask = np.all((flat_samples >= quantiles_low) & (flat_samples <= quantiles_high),axis=1)
    paras_subset_1sigma = flat_samples[mask]
    chi2_subset_1sigma = chi2_samples[mask]
    mask1=np.argmin(chi2_subset_1sigma)
    chi2_min = np.min(chi2_subset_1sigma)
    theta_best=paras_subset_1sigma[mask1]
    print(chi2_min)
'''
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
    np.savetxt(tar_dir+'SYN_%spc_spec.txt'%i,filei)'''