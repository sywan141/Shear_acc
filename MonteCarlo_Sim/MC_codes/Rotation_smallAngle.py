import numpy as np
import time
#import astropy.units as u
import astropy.constants as const
from numpy.random import default_rng
import multiprocessing
import os
import shutil
import yaml
from tqdm import tqdm

start_time = time.time()
m_p = (const.m_p.cgs).value
m_e = (const.m_e.cgs).value
c = (const.c.cgs).value
e = (const.e.esu).value
pi = np.pi
m_par = m_e

output_dir = '/home/wsy/Acc_MC/Results/trial_Rotation/'
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
N_bins = config['N_bins'] # 时间步长

jet_type = config['type']
syn = config['SYN_flag']
SA = config['SA_flag']
Sh = config['Shear_flag']


if jet_type=='kolgv': 
    q = 5/3
elif jet_type=='Bohm':
    q = 1
elif jet_type=='HS':
    q = 2
else:
    raise ValueError("The input should be meaningful")

# jet速度谱
def beta_dis(r):
    beta_max = np.sqrt(1-1/GM0**2)
    r1 = eta*R_sh
    if (r < r1):
        return beta_max
    elif (r > R_sh):
        return 0.0
    else: 
        return beta_max-(beta_max-beta_min)/(R_sh-r1)*(r-r1)

# 获取任意向量天顶角和方位角    
def Get_Angles(u):
    ux,uy,uz = u
    norm = np.linalg.norm(u)
    if norm ==0:
        raise ValueError("The input should be a non-zero vector")
    theta = np.arccos(uz/norm)
    phi = np.arctan2(uy, ux)%(2*pi)
    return theta, phi

# 能量的lorentz变换
def LorentzGamma(beta1, beta2, theta, g_me): 
    dbeta = (beta2 - beta1) / (1 - beta1 * beta2)
    costheta = np.cos(theta)
    if (dbeta == 0):
        return g_me
    betae =  np.sqrt(1-1/g_me**2)
    dGM = 1.0 / np.sqrt(1 - dbeta**2)
    g_me2 = dGM * g_me * (1 - betae * dbeta * costheta)
    return g_me2

# 将时间转换到观测者系中
def LorentzT(beta,dz,dt):
    GM = 1/np.sqrt(1-beta**2)
    return GM*(dt + beta/c*dz)

# 位移函数
def movement_e(u_cmv,beta,dt, x, y, z):
    ux,uy,uz = u_cmv
    dx = ux*c*dt
    dy = uy*c*dt
    dz = 1/np.sqrt(1-beta**2)*c*(beta + uz)*dt
    dz_jet = uz*c*dt
    x += dx
    y += dy
    z += dz
    return x,y,z,dz_jet

# 坐标系间的角度变换
def Angle_Trans(beta1, beta2, theta, phi):
    dbeta = (beta2 - beta1) / (1 - beta1 * beta2)
    costheta = np.cos(theta)
    theta_new = np.arccos((costheta-dbeta)/(1-dbeta*costheta))
    return theta_new, phi

# 坐标变换矩阵
def Rot_mat(direct, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    if direct == 'x':
        Rx = np.array([[1.0, 0.0, 0.0],
                      [0.0, cosa, -sina],
                      [0.0, sina, cosa]],dtype=np.float128)
        return Rx
    elif direct == 'y':
        Ry = np.array([[cosa, 0.0, sina],
                      [0.0, 1.0, 0.0],
                      [-sina, 0.0, cosa]], dtype=np.float128)
        return Ry
    elif direct == 'z':
        Rz = np.array([[cosa, -sina, 0.0],
                      [sina , cosa, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float128)
        return Rz
    
# 在不考虑随机加速时，绕任意轴进行一次小角度旋转
def small_angle_scatter(u_cmv, dt, tau):
    rng = default_rng()  # 随机数生成器
    sig_theta = np.sqrt(dt/tau)
    rot_angle = rng.normal(0, sig_theta) # 生成随机绕轴旋转角度
    k = np.random.randn(3) # 生成随机旋转轴
    k = k/np.linalg.norm(k) # 单位矢量
    u_out = u_cmv*np.cos(rot_angle) + np.cross(k,u_cmv)*np.sin(rot_angle) + (1-np.cos(rot_angle))*np.dot(k,u_cmv)*k
    return u_out

# 随机加速情况, 在Alfven波参考系中进行一次小角度散射，再重新将速度转换到喷流随动系中
def SA_scatter(u_cmv, ba, dt, tau):
    k = np.random.randn(3) # 随机波矢方向
    k = k/np.linalg.norm(k) # 单位矢量
    theta, phi = Get_Angles(k)
    u_wav = Rot_mat('y', theta)@(Rot_mat('z', phi)@u_cmv) # 方向变换到静止Alfven波随动系
    uk_x, uk_y, uk_z = u_wav
    gamma_k = 1/(np.sqrt(1 - ba**2))
    
    # 考虑Alfven波的运动，进行Lorentz变换
    ukk_x = uk_x/(gamma_k*(1 - ba*uk_z))
    ukk_y = uk_y/(gamma_k*(1 - ba*uk_z))
    ukk_z = (uk_z - ba)/(1 - ba*uk_z)
    # 在波矢坐标系下旋转
    u_kk = np.array([ukk_x, ukk_y, ukk_z], dtype=np.float128)
    u_wav_out = small_angle_scatter(u_kk, dt, tau)
    uk_x_out, uk_y_out, uk_z_out = u_wav_out
    # 变换到静止Alfven波随动系
    ukx_out = uk_x_out/(gamma_k*(1 + ba*uk_z_out))
    uky_out = uk_y_out/(gamma_k*(1 + ba*uk_z_out))
    ukz_out = (uk_z_out + ba)/(1 + ba*uk_z_out)
    uk_out = np.array([ukx_out, uky_out, ukz_out], dtype=np.float128)
    # 变换到喷流随动系
    uj_out = Rot_mat('z', -phi)@(Rot_mat('y', -theta)@uk_out)
    norm_uj_out = np.linalg.norm(uj_out)
    if norm_uj_out >= 1:
        uj_out = uj_out / norm_uj_out * 0.999999
    return uj_out

def Single_Par(K):
    
    # 初始参量（保持不变）
    rg = np.sqrt(g_me0**2 - 1) * m_e * c**2 / e / B0
    ba = B0 / np.sqrt(B0**2 + 4 * pi * n_p * m_p * c**2)  # 无量纲Alfven波速
    Dpp = xi * ba**2 / (1 - ba**2) * g_me0**2 * c / rg**(2 - q) / Lam_max**(q - 1)
    tacc = g_me0**2 / Dpp  # 加速时标
    tau = (rg)**(2-q) * Lam_max**(q-1) * (c * xi)**(-1)  # 散射时标
    
    theta_prev = 0
    phi_prev = 0
    theta = theta_prev
    
    beta1 = beta_dis(r0)  # 初始流速
    beta2 = beta_dis(r0)
    beta_ini = 0
    gme_obs = LorentzGamma(beta1, beta_ini, -1, g_me0) # 时间转换
    
    t_j = 0
    t_o = 0
    dt = tau / N_bins  # 初始时间步长
    g_me = g_me0
    r = r0
    [x, y, z] = [r0, 0, 0]  # 初始位置
    
    # 数组初始化
    gme_jetL = np.zeros(N_time + 2)
    r_jetL = np.zeros(N_time + 2)
    gme_obsL = np.zeros(N_time + 2)
    t_jetL = np.zeros(N_time + 2)
    t_obsL = np.zeros(N_time + 2)
    x_jetL = np.zeros(N_time + 2)
    y_jetL = np.zeros(N_time + 2)
    z_jetL = np.zeros(N_time + 2)
    
    gme_jetL[0] = g_me
    r_jetL[0] = r
    gme_obsL[0] = gme_obs
    t_jetL[0] = t_j
    t_obsL[0] = t_o
    x_jetL[0] = r0
    y_jetL[0] = 0
    z_jetL[0] = 0
    
    N_count = N_time
    

    while N_count >= 0:
        
        
        # 如果考虑剪切加速，则改变粒子速度
        if Sh:
            g_me = LorentzGamma(beta1, beta2, theta, g_me)
        u_cmv0 = np.sqrt(1-1/g_me**2) # 共动系中速度大小不变
        u_cmv = u_cmv0*np.array([np.sin(theta_prev)*np.cos(phi_prev),
                                 np.sin(theta_prev)*np.sin(phi_prev), 
                                 np.cos(theta_prev)], dtype = np.float128) # 共动系中初始方向因为坐标系变换改变
        
        # 发生SA时，方向改变
        if SA:
            u_cmv = SA_scatter(u_cmv, ba, dt, tau)
            u_norm = np.linalg.norm(u_cmv)
            g_me = 1/(np.sqrt(1-u_norm**2)) # SA能量变化
            theta, phi = Get_Angles(u_cmv) # 出射速度方向
        else:
            u_cmv = small_angle_scatter(u_cmv, dt, tau) # 共动系内弹性散射，速度大小不变
            theta, phi = Get_Angles(u_cmv) # 出射速度方向
            
        
        # 是否需要速度变化？
        beta1 = beta_dis(r)
        x,y,z,dz_jet = movement_e(u_cmv, beta2 , dt, x, y, z)
        r = np.sqrt(x**2 + y**2)
        
        if syn:
            g_me -= 1.1e-15 * g_me**2 * B0**2 / m_par / c**2 * dt
        
        if r > R_sh:
            gme_jetL[N_time - N_count + 1:] = -1
            r_jetL[N_time - N_count + 1:] = -1
            gme_obsL[N_time - N_count + 1:] = -1
            t_jetL[N_time - N_count + 1:] = -1
            t_obsL[N_time - N_count + 1:] = -1
            x_jetL[N_time - N_count + 1:] = 10 * R_sh
            y_jetL[N_time - N_count + 1:] = 10 * R_sh
            z_jetL[N_time - N_count + 1:] = -1
            break
        
        # 更新参数
        t_j += dt
        t_o += LorentzT(beta2, dz_jet, dt)
        rg = np.sqrt(g_me**2 - 1) * m_e * c**2 / e / B0
        tau = (rg)**(2 - q) * Lam_max**(q - 1) * (c * xi)**(-1)
        #dt = tau / N_bins
        gme_obs = LorentzGamma(beta2, beta_ini, -1, g_me)
        beta2 = beta_dis(r) # 最后一步再更新beta2
        theta_prev, phi_prev = Angle_Trans(beta1, beta2, theta, phi) # 只改变下一步粒子在共动系中的初始方向
        
        # 直接进行速度-能量变换结果如何？
        
        # 记录结果        
        t_jetL[N_time - N_count + 1] = t_j
        t_obsL[N_time - N_count + 1] = t_o
        gme_jetL[N_time - N_count + 1] = g_me
        r_jetL[N_time - N_count + 1] = r
        gme_obsL[N_time - N_count + 1] = gme_obs
        x_jetL[N_time - N_count + 1] = x
        y_jetL[N_time - N_count + 1] = y
        z_jetL[N_time - N_count + 1] = z
        
        N_count -= 1
    
    return t_jetL, t_obsL, gme_jetL, gme_obsL, r_jetL, x_jetL, y_jetL, z_jetL


if __name__ == '__main__':
    pool = multiprocessing.Pool(1000)
    numbers = range(N_par)
    results = []
    with tqdm(total=N_par, desc="Processing particles") as pbar:
        for result in pool.imap_unordered(Single_Par, numbers):
            results.append(result)
            pbar.update(1)
    pool.close()
    pool.join()
    
    t_jetL = np.array([res[0] for res in results])
    t_obsL = np.array([res[1] for res in results])
    gme_jetL = np.array([res[2] for res in results])
    gme_obsL = np.array([res[3] for res in results])
    r_jetL = np.array([res[4] for res in results])
    x_jetL = np.array([res[5] for res in results])
    y_jetL = np.array([res[6] for res in results])
    z_jetL = np.array([res[7] for res in results])
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "t_jetL.npy"), t_jetL)
    np.save(os.path.join(output_dir, "t_obsL.npy"), t_obsL)
    np.save(os.path.join(output_dir, "gme_jetL.npy"), gme_jetL)
    np.save(os.path.join(output_dir, "r_jetL.npy"), r_jetL)
    np.save(os.path.join(output_dir, "gme_obsL.npy"), gme_obsL)
    np.save(os.path.join(output_dir, "x_jetL.npy"), x_jetL)
    np.save(os.path.join(output_dir, "y_jetL.npy"), y_jetL)
    np.save(os.path.join(output_dir, "z_jetL.npy"), z_jetL)