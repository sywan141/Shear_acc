import numpy as np
import astropy.constants as const
import yaml

m_e = (const.m_e.cgs).value
m_p = (const.m_p.cgs).value
c = (const.c.cgs).value
e = (const.e.esu).value
pi = np.pi
sigma_T = 8*pi/3*(e**2/(m_e*c**2))**2 # Thomson section

# parameters
out_dir = '/home/wsy/Acc_MC/num_sols/results/'
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
Nu = int(config['N_grids']) # 格点数

jet_type = str(config['type'])
syn = bool(config['SYN_flag'])
SA = bool(config['SA_flag'])
Sh = bool(config['Shear_flag'])
ESC = bool(config['ESC_flag'])

syn_flag = 1
esc_flag = 1
# 不考虑同步辐射
if syn == False:
    syn_flag = 0
# 不考虑逃逸
if ESC == False:
    esc_flag = 0
    

if jet_type=='kolgv': 
    q = 5/3
elif jet_type=='Bohm':
    q = 1
elif jet_type=='HS':
    q = 2
else:
    raise ValueError("The input should be meaningful")

# 速度谱
def beta_dis(r):
    beta_max = np.sqrt(1-1/GM0**2)
    r1 = eta*R_sh
    if (r < r1):
        return beta_max
    elif (r > R_sh):
        return 0.0
    else: 
        return beta_max-(beta_max-beta_min)/(R_sh-r1)*(r-r1)
    
# scattering timescale
def tau(g_me):
    rg = g_me*m_e*c**2/(e*B0)
    tau = rg**(2-q)*(c*xi)**(-1)*Lam_max**(q-1)
    return tau

tau0 = tau(g_me0) # 初始散射时标
x_min, x_max = 0,12 # 对数格点
rg0 = g_me0*m_e*c**2/(e*B0)
Ba = B0 / np.sqrt(B0**2 + 4 * pi * n_p * m_p * c**2) # 相对喷流随动系的Alfven波波速
Dpp0 = xi * Ba**2 / (1 - Ba**2) * g_me0**2 * c / rg0**(2 - q) / Lam_max**(q - 1) # Dpp
tacc0 = g_me0**2 / Dpp0
dt = tacc0/N_bins # 时间步长

x = np.linspace(x_min,x_max,Nu) # x空间上的均匀网格
dx = (x_max - x_min)/(Nu - 1) # 步长
gamma = 10**x # 转换到能量
t_steps = N_time # 总步数
#print(gamma)

# 方程系数
def coeffs(gme):
    beta0 = beta_dis(0)
    beta_inj = beta_dis(r0)
    Gamma_j4 = 1/(1-beta_inj**2)**2 # 只与注入层相关
    Grad = beta0/((1-eta)*R_sh)
    
    # SA相关
    #Ba = B0/(np.sqrt(B0**2 + 4*pi*n_p*m_p*c**2))
    Gma4 = 1/(1-Ba**2)**2
    
    # 通用系数
    gamma_c = -sigma_T * B0**2 * gme**2 / (6 * pi * m_e * c)*syn_flag # 冷却项
    d_gamma_c = -sigma_T*B0**2*gme/(3*pi*m_e*c)* syn_flag
    t_esc = 3*R_sh**2/(c**2*tau(gme)) # diffusive逃逸时标
    
    
    if not SA:  # shear only
        Coef_sh = (2/15)*Gamma_j4*Grad**2*(c*gme)**2*tau(gme) # 漂移系数
        d_coef_sh = (4-q)*Coef_sh/gme # 系数微分
        phi_gme = 0.5*Coef_sh
        p_gme = 0.5*d_coef_sh-(1/gme)*Coef_sh-gamma_c
        q_gme = (1/gme**2)*Coef_sh-(1/gme)*d_coef_sh - d_gamma_c - 1/t_esc*esc_flag
    elif not Sh: # stochastic only
        Coef_sa = xi*Gma4*Ba**2*c*Lam_max**(1-q)*(m_e*c**2/(e*B0))**(q-2)*gme**q
        d_coef_sa = q*Coef_sa/gme
        phi_gme = 0.5*Coef_sa
        p_gme = 0.5*d_coef_sa-(1/gme)*Coef_sa-gamma_c
        q_gme = (1/gme**2)*Coef_sa-(1/gme)*d_coef_sa - d_gamma_c - 1/t_esc*esc_flag
    else: # total contribution
        Coef_sh = (2/15)*Gamma_j4*Grad**2*(c*gme)**2*tau(gme) # 漂移系数
        d_coef_sh = (4-q)*Coef_sh/gme # 系数微分
        Coef_sa = xi*Gma4*Ba**2*c*Lam_max**(1-q)*(m_e*c**2/(e*B0))**(q-2)*gme**q
        d_coef_sa = q*Coef_sa/gme
        Coef_tot = Coef_sh + Coef_sa
        d_coef_tot = d_coef_sh + d_coef_sa
        phi_gme = 0.5*Coef_tot
        p_gme = 0.5*d_coef_tot-(1/gme)*Coef_tot-gamma_c
        q_gme = (1/gme**2)*Coef_tot-(1/gme)*d_coef_tot - d_gamma_c - 1/t_esc*esc_flag
        
    
    return phi_gme, p_gme, q_gme

# 中心差分函数
def N_prime(N,i):
    if i==0 or i==Nu-1:
        return 0
    return (N[i+1]-N[i-1])/(2*dx)

def N_prime_p(N,i):
    if i==0 or i==Nu-1:
        return 0
    return (N[i+1]-2*N[i]+N[i-1])/(dx**2)

phi_gm, p_gm, q_gm = coeffs(gamma)
Q = np.exp(-100*(np.log(gamma) - np.log(g_me0))**2) # 源项

# CFL条件判断
print("Max phi/gamma^2:", np.max(phi_gm / gamma**2))
print("Max |p/gamma|:", np.max(np.abs(p_gm / gamma)))
cfl_diff = dx**2 / (2 * np.max(phi_gm / gamma**2))
cfl_conv = dx / np.max(np.abs(p_gm / gamma))
print("CFL diffusion limit:", cfl_diff)
print("CFL convection limit:", cfl_conv)
print("Current dt:", dt)

N = Q # 初始条件
N_history = []

for step in range(t_steps):
    N_history.append(N)
    N_temp = N.copy() # 每次循环都需要初始化一次，否则只会保留最后一次的值
    for i in range(1, Nu-1):
        N_dot = ((phi_gm[i]/gamma[i]**2)*(N_prime_p(N,i)-N_prime(N,i))+
                 (p_gm[i]/gamma[i])*N_prime(N,i)+
                 q_gm[i]*N[i]+
                 Q[i])
        N_temp[i] = N[i] + dt*N_dot
        
    # 边界条件
    N_temp[0] = 0
    N_temp[-1] = 0
    N = N_temp

#print(N_history)
np.savetxt(out_dir+'solutions.txt', N_history)