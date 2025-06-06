import numpy as np
from scipy.special import jv
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool
import datetime
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import simpson

start = datetime.datetime.now()
# Constants setup
c = 1 #const.c.cgs.value
pi = np.pi
q = 7/3
Lam_max = 1e18
k_min = 2 * pi / Lam_max
B0 = ((1e4 * u.uG).to(u.G)).value
eta = 0.01
beta_A = 0.9  # Ensure this is dimensionless
C_g = (q - 1) * (3 * q - 5) / (16 * pi * k_min**3)
minor = 1e-3
thisdir = '/home/wsy/V4641_Sgr/FP_coefs/'

# Goldreich-Sridhar spectrum
def GS_spec(k, mu_k):
    x = (k / k_min)**(1/3) * (mu_k / (1 - mu_k**2)**(1/3))
    #g = np.heaviside(1 - np.abs(x), 0.5)
    g = 0.5 * (1 + np.tanh((1 - np.abs(x)) / 0.1)) # to replace the heaviside function with a smooth one
    S_k = (eta / (1 - eta)) * B0**2 * (k * np.sqrt(1 - mu_k**2) / k_min)**(-q - 1) * g *C_g
    return S_k


# n=1 or -1, gyroresonance(Landau)
def integrand_D(n, r_g, MU, MU_K, sign):
    Omega = c / r_g
    mu, mu_k = MU, MU_K
    
    # Compute A/B based on sign
    if sign == 'plus':
        AB = np.abs(mu * mu_k*c - beta_A * c * mu_k)
    else:
        AB = np.abs(mu * mu_k*c + beta_A * c * mu_k)
    
    k_val = -(n * Omega) / AB
    mask = (AB != 0) & (k_val >= k_min)
    
    z = np.zeros_like(AB)
    #valid = mask
    sqrt_term = np.sqrt(1 - mu_k**2) * np.sqrt(1 - mu**2)
    z[mask] = (r_g * k_val[mask] * sqrt_term[mask]) / c
    
    Jv = (jv(n + 1, z) + jv(n - 1, z))**2
    c1 = (Omega**2 * pi**2 * (1 - mu**2)) / (2 * B0**2)
    
    if sign == 'plus':
        term = (c - mu * beta_A * c)**2
    else:
        term = (c + mu * beta_A * c)**2
    
    spec = GS_spec(k_val, mu_k)
    result = c1 * k_val**2 * Jv * term * spec
    return np.where(mask, result, 0.0) # np.where(A,B,C): if A then B, or C

def Total_D(r_g):
    mu_min, mu_max = -1+minor, 1-minor # to avoid 0
    mu_k_min, mu_k_max = -1+minor, 1-minor
    n_x, n_y = 7000,7000  # Grid resolution

    # Generate integration grid to avoid loops
    mu = np.linspace(mu_min, mu_max, n_x, endpoint=False)
    mu_k = np.linspace(mu_k_min, mu_k_max, n_y, endpoint=False)
    MU, MU_K = np.meshgrid(mu, mu_k, indexing='ij')
    
    # Grid spacing
    delta_mu = (mu_max - mu_min) / n_x
    delta_mu_k = (mu_k_max - mu_k_min) / n_y

    # Compute all terms
    n_min = np.ceil(r_g*k_min)
    n_all = np.arange(-30,30)
    n_valid = n_all[np.abs(n_all)>n_min]
    terms = []
    for n_i in n_valid:
        terms.append(integrand_D(n_i, r_g, MU, MU_K, 'plus'))
        terms.append(integrand_D(n_i, r_g, MU, MU_K, 'minus'))
    '''
    terms = [
        integrand_D(1, r_g, MU, MU_K, 'plus'),
        integrand_D(-1, r_g, MU, MU_K, 'plus'),
        integrand_D(1, r_g, MU, MU_K, 'minus'),
        integrand_D(-1, r_g, MU, MU_K, 'minus')
    ]'''

    integral = sum(np.sum(term) for term in terms) * delta_mu * delta_mu_k # trangular integration
    return integral

# Parallel computation setup
if __name__ == '__main__':
    r_g = np.logspace(-4, 2, 30) / k_min
    with Pool(processes=20) as pool:
        D_res = pool.map(Total_D, r_g)
    
    D_res = np.array(D_res) / (c * k_min)  # normalization

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title(r'Fokker-Planck Coefficient')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(r_g * k_min, D_res)
    plt.scatter(r_g * k_min, D_res)
    plt.xlabel(r'$r_g k_{\rm min}$')
    plt.ylabel(r'$D_{\mu \mu}$')
    plt.xlim(1e-4,1e1)
    plt.grid(True, which='both', linestyle='--')
    plt.savefig(thisdir + 'D_betaA=%s_q=%s.png'%(beta_A,q))
    plt.show()
end = datetime.datetime.now()
print('Total time:%s'%(end-start))