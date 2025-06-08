# 只考虑发生在喷流横截面上的位移，只有少数散射事件发生
import numpy as np
import astropy.constants as const
import multiprocessing
import os
import shutil
import yaml
import time
from numpy.random import default_rng
from tqdm import tqdm
from mpmath import mp

start_time = time.time()
m_p = (const.m_p.cgs).value
m_e = (const.m_e.cgs).value
c = (const.c.cgs).value
e = (const.e.esu).value
pi = np.pi
m_par = m_e
nan = np.nan
mp.dps = 30

output_dir = '/home/wsy/Acc_MC/MC_sim/codes/Code_Trials/Results/'
# 读取初始参数
with open('/home/wsy/Acc_MC/MC_sim/paras.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
R_sh = float(config['R_sh'])
GM0 = float(config['GM0'])
eta = float(config['eta'])
beta_min = float(config['beta_min'])
B0 = float(config['B0'])
xi = float(config['xi'])
Lam_max = float(config['Lam_max'])
g_me0 = float(config['g_me0']) 
r0 = float(config['r0'])
n_p = float(config['n_p'])

N_par = int(config['N_par']) # 粒子数
N_time = int(float((config['N_time']))) # 时间步数
N_bins = int(config['N_bins']) # 时间步长

jet_type = str(config['type'])
syn = bool(config['SYN_flag'])
SA = bool(config['SA_flag'])
Sh = bool(config['Shear_flag'])
ESC = bool(config['ESC_flag'])

gmes = np.logspace(1,9, 100)

if jet_type=='kolgv': 
    q = 5/3
elif jet_type=='Bohm':
    q = 1
elif jet_type=='HS':
    q = 2
else:
    raise ValueError("The input should be meaningful")


ba = B0 / np.sqrt(B0**2 + 4 * pi * n_p * m_p * c**2)  # 无量纲Alfven波速


# jet速度谱
def beta_dis(r):
    beta_max = np.sqrt(1-1/GM0**2)
    r1 = eta*R_sh
    if (r < r1):
        return beta_max
    elif (r > R_sh):
        return 0.0
    else: 
        return mp.mpf(beta_max-(beta_max-beta_min)/(R_sh-r1)*(r-r1))
    
def Rg_calc(g_me):
    return mp.mpf(np.sqrt(g_me**2)*m_e*c**2/(e*B0))

def tau_calc(g_me):
    return mp.mpf(Rg_calc(g_me)**(2-q)*Lam_max**(q-1)*(c*xi)**(-1))
    
def movement_e(g_me,costheta, phi, dt, x, y):
    u_cmv = mp.mpf(np.sqrt(1-1/g_me**2))
    sintheta = np.sqrt(1-costheta**2)
    
    ux = u_cmv*sintheta*np.cos(phi)
    uy = u_cmv*sintheta*np.sin(phi)
    #uz = u_cmv*costheta
    
    dx = ux*c*dt
    dy = uy*c*dt
    #dz_jet = uz*c*dt
    x += dx
    y += dy
    
    return x,y

def LorentzGamma(beta1, beta2, costheta, g_me): 
    dbeta = mp.mpf((beta2 - beta1) / (1 - beta1 * beta2))
    #print(dbeta)
    if (dbeta == 0):
        return g_me
    betae =  mp.mpf(np.sqrt(1-1/g_me**2))
    dGM = mp.mpf(1.0 / np.sqrt(1 - dbeta**2))
    #print(dGM)
    g_me2 = mp.mpf(dGM * g_me * (1 - betae * dbeta * costheta))
    #print(-dbeta*costheta)
    g_me2 = mp.mpf(g_me*(1 + 0.5*dbeta**2 - betae*dbeta*costheta))
    #if (dGM*(1 - betae * dbeta * costheta)<=1):
        #print(dGM*(1 - betae * dbeta * costheta))
    return g_me2


def Single_Par(args):
    
    K, seed = args  # 传入种子
    rng = default_rng(seed) # 生成每个粒子的角度分布
    
    # 初始参量（保持不变）
    beta_ini = beta_dis(r0)
    G_res = np.zeros(len(gmes))
    cos_res = np.zeros(len(gmes))
    phis = np.zeros(len(gmes))
    alphas = np.zeros(len(gmes))
    beta_D = np.zeros(len(gmes))
    
    
    for i in range(len(gmes)):
        
        # 初始化粒子参数
        gme = mp.mpf(gmes[i])
        #rg = Rg_calc(gme)
        tau = tau_calc(gme)
        #tau0 = tau_calc(100)
        
        costheta = 2*rng.random() - 1 #2*rng.random() - 1  # 均匀分布的极角
        
        phi = 2*pi*rng.random()  # 均匀分布的方位角
        
        alpha = 2*pi*rng.random()  # 均匀分布的方位角
        
        r_tmp = r0
        [x,y] = [r0*np.cos(alpha),r0*np.sin(alpha)]
        
        dt = tau/N_bins # 步长
        cos_res[i] = costheta
        phis[i] = phi
        alphas[i] = alpha
        
        for Nth in range(N_bins):
            x,y = movement_e(gme,costheta, phi, dt, x, y)
            r_tmp = mp.mpf(np.sqrt(x**2 + y**2))
            beta_tmp = beta_dis(r_tmp)
            poss = np.random.uniform(0,1)
            
            if r_tmp > R_sh:
                gme = nan
                print('The particle moves out of the jet')
                break
          
            #if poss > 1-np.exp(-dt/tau):
                #continue
            #else:
            gme = LorentzGamma(beta_ini, beta_tmp, costheta, gme)
            beta_D[i] = mp.mpf((beta_tmp - beta_ini) / (1 - beta_tmp * beta_ini))
            tau = tau_calc(gme)
            # 各向同性散射
            costheta = 2*rng.random() - 1
            #print(costheta)
            phi = 2 * pi * rng.random()
            # 返回中心
            r_tmp = r0
            x = r0
            y = 0  
            #print(gme)
        G_res[i] = gme
    
    return G_res, cos_res, phis, alphas, beta_D
                


if __name__ == '__main__':
    pool = multiprocessing.Pool(64)
    
    # 全局随机数种子
    global_seed = int(1234)
    seed_rng = default_rng(global_seed)
    seeds = seed_rng.integers(0, 2**32, size=N_par)  # 为每个粒子生成一个种子
    
    numbers = [(K, seeds[K]) for K in range(N_par)]
    results = []
    
    
    with tqdm(total=N_par, desc="Processing particles") as pbar:
        for result in pool.imap_unordered(Single_Par, numbers):
            results.append(result)
            pbar.update(1)
    pool.close()
    pool.join()
    
    gammas = np.array([res[0] for res in results])
    cos_thetas = np.array([res[1] for res in results])
    ang_phis = np.array([res[2] for res in results])
    ang_alphas = np.array([res[3] for res in results])
    beta_Ds = np.array([res[4] for res in results])
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "Gammas.npy"), gammas)
    np.save(os.path.join(output_dir, "Coses.npy"), cos_thetas)
    np.save(os.path.join(output_dir, "Phis.npy"), ang_phis)
    np.save(os.path.join(output_dir, "Alphas.npy"), ang_alphas)
    np.save(os.path.join(output_dir, "beta_D.npy"), beta_Ds)