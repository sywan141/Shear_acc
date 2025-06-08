import numpy as np
import time
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

output_dir = '/home/wsy/Acc_MC/Results/trial_Rotation_RW/'
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

# 将速度转换到新的共动系中
def Vel_Shear(beta1, beta2, u_cmv): 
    #epsilon = 1e-16
    ux, uy, uz = u_cmv
    dbeta = (beta2 - beta1) / (1 - beta1 * beta2)
    gm_dbeta = 1/(np.sqrt(1-dbeta**2))
    ux_prime = ux/(gm_dbeta*(1-dbeta*uz))
    uy_prime = uy/(gm_dbeta*(1-dbeta*uz))
    uz_prime = (uz-dbeta)/(1-dbeta*uz)
    u_prime = np.array([ux_prime ,uy_prime ,uz_prime], dtype=np.float128)
    #norm_u_prime = np.linalg.norm(u_prime)
    #print(gm_dbeta)
    #print(np.linalg.norm(u_prime)/np.linalg.norm(u_cmv))
    '''
    if (norm_u_prime>=1):
        u_prime = u_prime/norm_u_prime*0.99999999'''
    return u_prime

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
                      [-sina, 0.0, cosa]],dtype=np.float128)
        return Ry
    elif direct == 'z':
        Rz = np.array([[cosa, -sina, 0.0],
                      [sina , cosa, 0.0],
                      [0.0, 0.0, 1.0]],dtype=np.float128)
        return Rz
    
# 各向同性散射，生成随机方向的单位速度矢量
def isotropic_scatter(u_cmv):
    rng = default_rng()
    # 生成随机方向的单位矢量
    theta = np.arccos(2 * rng.random() - 1)  # 均匀分布的极角
    phi = 2 * pi * rng.random()  # 均匀分布的方位角
    u_out = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ],dtype=np.float128)
    # 保持速度大小不变
    u_norm = np.linalg.norm(u_cmv)
    u_out *= u_norm
    #print(np.linalg.norm(u_out))
    return u_out

# 随机加速情况，在Alfven波参考系中进行各向同性散射
def SA_scatter(u_cmv, ba):
    k = np.random.randn(3) # 随机波矢方向
    k = k/np.linalg.norm(k) # 单位矢量
    theta, phi = Get_Angles(k)
    u_wav = Rot_mat('y', theta)@(Rot_mat('z', phi)@u_cmv) # 方向变换到静止Alfven波随动系
    uk_x, uk_y, uk_z = u_wav
    gamma_k = 1/(np.sqrt(1 - ba**2))
    theta_in = Get_Angles(u_wav)[0]
    
    # 考虑Alfven波的运动，进行Lorentz变换
    ukk_x = uk_x/(gamma_k*(1 - ba*uk_z))
    ukk_y = uk_y/(gamma_k*(1 - ba*uk_z))
    ukk_z = (uk_z - ba)/(1 - ba*uk_z)
    # 在波矢坐标系下进行各向同性散射
    u_kk = np.array([ukk_x, ukk_y, ukk_z], dtype=np.float128)
    u_wav_out = isotropic_scatter(u_kk)
    uk_x_out, uk_y_out, uk_z_out = u_wav_out
    theta_out = Get_Angles(u_wav_out)[0]
    # 变换到静止Alfven波随动系
    ukx_out = uk_x_out/(gamma_k*(1 + ba*uk_z_out))
    uky_out = uk_y_out/(gamma_k*(1 + ba*uk_z_out))
    ukz_out = (uk_z_out + ba)/(1 + ba*uk_z_out)
    uk_out = np.array([ukx_out, uky_out, ukz_out],dtype=np.float128)
    # 变换到喷流随动系
    uj_out = Rot_mat('z', -phi)@(Rot_mat('y', -theta)@uk_out)
    #norm_uj_out = np.linalg.norm(uj_out)
    '''
    if norm_uj_out >= 1:
        uj_out = uj_out / norm_uj_out * 0.999999'''
    #print((np.sqrt(1-np.linalg.norm(u_cmv)**2))/(np.sqrt(1-np.linalg.norm(uj_out)**2)))
    return uj_out, theta_in, theta_out

# 能量变换
def LorentzGamma(beta1, beta2, theta, g_me): 
    costheta = np.cos(theta)
    dbeta = (beta2 - beta1) / (1 - beta1 * beta2)
    if (dbeta == 0):
        return g_me
    betae =  np.sqrt(1-1/g_me**2)
    dGM = 1.0 / np.sqrt(1 - dbeta**2)
    g_me2 = dGM * g_me * (1 - betae * dbeta * costheta)
    return g_me2

# 镜面反射，返回反射后的二维速度
def Relect(uxy, k):
    # 取模
    u_norm = np.linalg.norm(uxy)
    k_norm = np.linalg.norm(k) 
    
    norm_k = k/k_norm 
    costheta = np.dot(uxy,k)/(u_norm*k_norm)
    uy_re = - u_norm*costheta*norm_k # 径向速度取反向
    ux_re = uxy + uy_re
    uxy_re = ux_re + uy_re
    return uxy_re
    
def Single_Par(K):
    
    # 初始参量（保持不变）
    rg = np.sqrt(g_me0**2 - 1) * m_e * c**2 / e / B0
    ba = B0 / np.sqrt(B0**2 + 4 * pi * n_p * m_p * c**2)  # 无量纲Alfven波速
    Dpp = xi * ba**2 / (1 - ba**2) * g_me0**2 * c / rg**(2 - q) / Lam_max**(q - 1)
    tacc = g_me0**2 / Dpp  # 加速时标
    tau = (rg)**(2-q) * Lam_max**(q-1) * (c * xi)**(-1)  # 散射时标
    
    # 初始方向随机
    rng = default_rng()
    theta_prev = np.arccos(2 * rng.random() - 1)  # 均匀分布的极角
    phi_prev = 2 * pi * rng.random()  # 均匀分布的方位角

    #beta_sc = beta_dis(r0) # 散射位置标记 
    
    # 临时随机化注入位置
    #poss = 1e-4 #np.random.uniform(eta,1)
    #r0 = poss*R_sh   
    
    
    t_j = 0
    t_o = 0
    dt = tau / N_bins  # 初始时间步长
    g_me = g_me0
    r = r0
    [x, y, z] = [r0, 0, 0]  # 初始位置
    
    beta1 = beta_dis(r0)  # 上一层流速
    beta2 = beta_dis(r0)  # 下一层流速
    beta_ini = 0 # 静止系流速
    gme_obs = LorentzGamma(beta1, beta_ini, -1, g_me0) # 时间转换
    
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
        
        poss = np.random.uniform(0,1) # 判断是否发生散射
        if poss >  1- np.exp(-dt/tau): 
            
            u_cmv0 = np.sqrt(1-1/g_me**2) 
            u_cmv = u_cmv0*np.array([np.sin(theta_prev)*np.cos(phi_prev),
                                 np.sin(theta_prev)*np.sin(phi_prev), 
                                 np.cos(theta_prev)],dtype=np.float128) 
            
            # 考虑剪切加速效应时，更新一次能量
            if Sh:
                theta = Get_Angles(u_cmv)[0]
                g_me = LorentzGamma(beta1, beta2, theta, g_me) # 能量变化
            u_cmv = Vel_Shear(beta1, beta2, u_cmv) # 将速度变到现在的共动系中
            
            beta1 = beta_dis(r) # 记录位移前的流速
            x,y,z,dz_jet = movement_e(u_cmv, beta2 ,dt, x, y, z) 
            r = np.sqrt(x**2 + y**2)
            
            if syn:
                g_me -= 1.1e-15 * g_me**2 * B0**2 / m_par / c**2 * dt
            
            if (r > R_sh):
                if ESC:
                    gme_jetL[N_time-N_count+1:]=-1
                    r_jetL[N_time-N_count+1:]=-1
                    gme_obsL[N_time-N_count+1:]=-1
                    t_jetL[N_time-N_count+1:]=-1
                    t_obsL[N_time-N_count+1:]=-1
                    x_jetL[N_time-N_count+1:]= 10*R_sh
                    y_jetL[N_time-N_count+1:]= 10*R_sh
                    z_jetL[N_time-N_count+1:]= -1  
                    break
                # 不考虑逃逸时，粒子在边界层发生一次镜像反射
                else:
                    ux,uy, uz = u_cmv
                    k = np.array([x, y])
                    uxy = np.array([ux, uy])
                    uxy_re = Relect(uxy,k)
                    u_cmv = np.array([uxy_re[0], uxy_re[1], uz])
                    Ang = np.arctan2(y,x)
                    
                    # 重置粒子位置
                    r = R_sh
                    x = r*np.cos(Ang)
                    y = r*np.sin(Ang)
                    
            # 没有出界则更新参数
            t_j += dt
            t_o += LorentzT(beta2,dz_jet,dt)
            rg = np.sqrt(g_me**2 - 1) * m_e * c**2 / e / B0
            tau = (rg)**(2-q)*Lam_max**(q-1)*(c*xi)**(-1)
            #dt = tau/N_bins
            gme_obs = LorentzGamma(beta2, beta_ini, -1, g_me)
            beta2 = beta_dis(r) # 记录新的beta2, 即位移后的流速
            theta_prev, phi_prev = Get_Angles(u_cmv) # 记录本步速度的方向
            
            t_jetL[N_time-N_count+1] = t_j
            t_obsL[N_time-N_count+1] = t_o
            gme_jetL[N_time-N_count+1] = g_me
            r_jetL[N_time-N_count+1] = r
            gme_obsL[N_time- N_count +1] = gme_obs
            x_jetL[N_time-N_count+1] = x
            y_jetL[N_time-N_count+1] = y
            z_jetL[N_time-N_count+1] = z
            
            N_count-=1
            continue
        
        # 在散射分支下，先更新一次速度
        u_cmv0 = np.sqrt(1-1/g_me**2)
        u_cmv = u_cmv0*np.array([np.sin(theta_prev)*np.cos(phi_prev),
                                    np.sin(theta_prev)*np.sin(phi_prev), 
                                    np.cos(theta_prev)],dtype=np.float128) # 继承上一步的速度
        
        # 考虑剪切加速效应时，更新一次能量
        if Sh:
            theta = Get_Angles(u_cmv)[0]
            g_me = LorentzGamma(beta1, beta2, theta, g_me) # 能量变化, 角度值为在上一个坐标系的值
            u_cmv = Vel_Shear(beta1, beta2, u_cmv) # 将速度变到现在的共动系中
            
        if SA:
            # 若考虑随机加速，则出射方向只和波共动系下的散射相关
            u_cmv = SA_scatter(u_cmv, ba)[0]
            theta_in = SA_scatter(u_cmv, ba)[1]
            theta_out = SA_scatter(u_cmv, ba)[2]
            g_me = LorentzGamma(0, ba, theta_in, g_me)
            g_me = LorentzGamma(ba, 0, theta_out, g_me)
            theta, phi = Get_Angles(u_cmv) # 出射速度方向
        else:
            # 若不考虑SA，则随机指派一个出射方向
            u_cmv = isotropic_scatter(u_cmv)
            theta, phi = Get_Angles(u_cmv) # 出射速度方向
        
        if syn:
            g_me -= 1.1e-15 * g_me**2 * B0**2 / m_par / c**2 * dt
        
        # 是否需要速度变化？
        beta1 = beta_dis(r)
        #beta_sc = beta_dis(r)
        x,y,z,dz_jet = movement_e(u_cmv, beta2 , dt, x, y, z)
        r = np.sqrt(x**2 + y**2)
        
        if r > R_sh:
            if ESC:
                gme_jetL[N_time - N_count + 1:] = -1
                r_jetL[N_time - N_count + 1:] = -1
                gme_obsL[N_time - N_count + 1:] = -1
                t_jetL[N_time - N_count + 1:] = -1
                t_obsL[N_time - N_count + 1:] = -1
                x_jetL[N_time - N_count + 1:] = 10 * R_sh
                y_jetL[N_time - N_count + 1:] = 10 * R_sh
                z_jetL[N_time - N_count + 1:] = -1
                break
                            # 不考虑逃逸时，粒子在边界层发生一次镜像反射
            else:
                ux,uy, uz = u_cmv
                k = np.array([x, y])
                uxy = np.array([ux, uy])
                uxy_re = Relect(uxy,k)
                u_cmv = np.array([uxy_re[0], uxy_re[1], uz])
                Ang = np.arctan2(y,x)
                    
                
                # 重置粒子位置
                r = R_sh
                x = r*np.cos(Ang)
                y = r*np.sin(Ang)
         
        # 更新参数
        t_j += dt
        t_o += LorentzT(beta2, dz_jet, dt)
        rg = np.sqrt(g_me**2 - 1) * m_e * c**2 / e / B0
        tau = (rg)**(2 - q) * Lam_max**(q - 1) * (c * xi)**(-1)
        #dt = tau / N_bins
        gme_obs = LorentzGamma(beta2, beta_ini, -1, g_me)
        beta2 = beta_dis(r) # 最后一步再更新beta2
        theta_prev = theta
        phi_prev = phi
        
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
    pool = multiprocessing.Pool(64)
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