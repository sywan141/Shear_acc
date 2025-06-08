# 在没有散射发生时，累积能量；在散射发生时，将粒子拉回初始位置
# 关掉逃逸时，粒子在到达边界时，会被弹回初始位置
# 连续注入演化
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
nan=float(np.nan)
m_par = m_e

output_dir = '/home/wsy/Acc_MC/Results/trial_Cinjection_Drag/'
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
ESC = config['ESC_flag']

if jet_type=='kolgv': 
    q = 5/3
elif jet_type=='Bohm':
    q = 1
elif jet_type=='HS':
    q = 2
    

# 喷流截面速度分布
def beta_dis(r, GM_0, r_max):
    r1 = eta*r_max
    beta_max = np.sqrt(1-1/GM_0**2)
    if np.abs(r)<r1:
        return beta_max
    elif np.abs(r)>r_max:
        return 0.0
    else:
        return beta_max + ((beta_min-beta_max)/(r_max-r1))*(np.abs(r)-r1)

# 粒子能量Lorentz变换
def LorentzGamma(beta1, beta2, costheta, g_me): 
    dbeta = (beta2 - beta1) / (1 - beta1 * beta2)
    if (dbeta == 0):
        return g_me
    betae =  np.sqrt(1-1/g_me**2)
    dGM = 1.0 / np.sqrt(1 - dbeta**2)
    g_me2 = dGM * g_me * (1 - betae * dbeta * costheta)
    return g_me2

# 随动系时间变到BH系
def LorentzT(beta,dz,dt):
    GM = 1/np.sqrt(1-beta**2)
    return GM*(dt + beta/c*dz)

# 位移函数，dt为随动系下的时间
def movement_e(g_me,costheta,alpha,beta,dt,x,y,z):
    sintheta = np.sqrt(1-costheta**2)
    v = c*np.sqrt(1-1/g_me**2) # 共动系速度
    dx_jet = v*sintheta*np.cos(alpha)*dt
    dy_jet = v*sintheta*np.sin(alpha)*dt
    dz_jet = v*costheta*dt
    dx = dx_jet                    
    dy = dy_jet
    dz = 1/np.sqrt(1-beta**2)*c*(beta + np.sqrt(1-1/g_me**2)*costheta)*dt
    x+=dx
    y+=dy
    z+=dz
    return x,y,z,dz_jet


# 角度转换
def Angle_Trans(beta1, beta2, costheta,alpha):
    dbeta = (beta2 - beta1) / (1 - beta1 * beta2)
    return (costheta-dbeta)/(1-dbeta*costheta), alpha


# 计算下一步粒子在喷流共动系下的位移方向
def mag_rotateT(costheta1, alpha1, costhetaM, sinthetaM, alphaM):
    sintheta1 = np.sqrt(1-costheta1**2)
    
    costheta2 = costhetaM*costheta1-sinthetaM*sintheta1
    sintheta2 = costhetaM*sintheta1+sinthetaM*costheta1
    
    cosa = costheta1**2 + sintheta1**2*np.cos(alpha1)
    sina = np.sqrt(1 - cosa**2)
    
    # a=0, 在喷流共动系中沿着磁云轴出射
    if (sina == 0):
        return costheta2, alphaM
    
    sinB = sintheta1*np.sin(alpha1)/sina 
    cosB = (costheta1 - costheta1 * cosa) / (sintheta1*sina)
    costheta=costheta2*cosa+sintheta2*sina*cosB
    sintheta=np.sqrt(1-costheta**2)
    
    sindalp = sinB * sina / sintheta if sintheta != 0 else 0 # 若直接沿喷流轴向出射，则方位角无意义，设为0
    cosdalp = (cosa - costheta2 * costheta) / (sintheta2 * sintheta) if sintheta2 * sintheta != 0 else 1
    
    # 保留方位角的反三角函数值, 得到出射方位角
    delta_alpha = np.arctan2(sindalp, cosdalp)
    alpha = (alphaM + delta_alpha) % (2 * np.pi)

    return costheta, alpha


def SA_Gamma(costheta_prev, alpha_prev, g_me, ba):
    # 生成随机的磁云方向
    rng = default_rng()
    rndm_cos = rng.random()
    costhetaM = 2 * rndm_cos - 1
    sinthetaM = np.sqrt(1 - costhetaM**2)
    rndm_alp = rng.random()
    alphaM = 2 * pi* rndm_alp

    sintheta_prev = np.sqrt(1 - costheta_prev**2)
    alpha_in = alphaM - alpha_prev
    cos_in = costheta_prev * costhetaM + sintheta_prev * sinthetaM * np.cos(alpha_in)
    
    # 假设在磁云共动系下发生了各向同性散射，出射方向随机
    rndm_o1 = rng.random()
    cos_out = 2 * rndm_o1 - 1
    rndm_o2 = rng.random()
    alpha_out = 2*pi*rndm_o2
    
    # 能量变化，只受坐标变换影响
    betae = np.sqrt(1-1/g_me**2)
    g_me = g_me*(1/(1-ba**2))*(1-betae*ba*cos_in)*(1+betae*ba*cos_out)
    
    # 计算从磁云系到喷流共动系下的出射角
    costheta1, alpha1 = Angle_Trans(ba, 0, cos_out,alpha_out)
    costheta, alpha = mag_rotateT(costheta1, alpha1, costhetaM, sinthetaM, alphaM) # 运动方向
    
    #print((1/(1-ba**2))*(1-betae*ba*cos_in)*(1+betae*ba*cos_out))
    return costheta, alpha, g_me
    
# 粒子演化函数
def Single_Par(K):
    rng = default_rng() # 随机数生成器
    # 初始参量
    rg = np.sqrt(g_me0**2 - 1) * m_e * c**2 / e / B0
    ba = B0 / np.sqrt(B0**2 + 4 * pi * n_p * m_p * c**2) # 相对喷流随动系的Alfven波波速
    Dpp = xi * ba**2 / (1 - ba**2) * g_me0**2 * c / rg**(2 - q) / Lam_max**(q - 1) # Dpp
    tacc = g_me0**2 / Dpp  # SA加速时标
    tau = (rg)**(2-q)*Lam_max**(q-1)*(c*xi)**(-1) # 散射时标
    
    # 初始方向随机
    rng = default_rng()
    costheta_prev = 2 * rng.random() - 1  # 均匀分布的极角
    alpha_prev = 2 * pi * rng.random()  # 均匀分布的方位角
    
    '''
    # 临时随机化注入位置
    poss = 1e-2 #np.random.uniform(0,1)
    r0 = poss*R_sh   
    '''
    
    beta2 = beta_dis(r0, GM0, R_sh) # 辐射时当前位置
    beta_ini = beta_dis(r0, GM0, R_sh) # 初始位置
    beta1 = beta_dis(r0, GM0, R_sh)
    
    # 喷流参考系和观测者系的计时起点相同
    dt = tau/N_bins # jet参考系下
    t_j = K*dt # 初始化注入时间
    g_me = g_me0
    r = r0
    [x,y,z] = [r0,0,0] # 初始位置
    
    gme_jetL = np.zeros(N_time+2)
    r_jetL = np.zeros(N_time+2)
    t_jetL = np.zeros(N_time+2)
    x_jetL = np.zeros(N_time+2)
    y_jetL = np.zeros(N_time+2)
    z_jetL = np.zeros(N_time+2)
    
    gme_jetL[K] = g_me
    r_jetL[K] = r
    t_jetL[K] = t_j
    x_jetL[K] = r0
    y_jetL[K] = 0
    z_jetL[K] = 0
    
    # 初始化进入演化前的参数
    if K >= 1:
        gme_jetL[0:K] = nan
        r_jetL[0:K] = nan
        t_jetL[0:K] = nan
        x_jetL[0:K] = nan
        y_jetL[0:K] = nan
        z_jetL[0:K] = nan

    N_count = N_time-K
    
    while (N_count>=0):
        poss = np.random.uniform(0,1) # 判断是否发生散射
        if poss >  1- np.exp(-dt/tau): # 无散射分支
            # 同步辐射能损
            if syn:
                g_me -= 1.1e-15 * g_me**2 * B0**2 / m_par / c**2 * dt
            
            # 在进行位移时，需要考虑角度变化
            #costheta_prev, alpha_prev = Angle_Trans(beta1, beta2, costheta_prev,alpha_prev) # 将方向修正到现有坐标系
            beta1 = beta_dis(r, GM0, R_sh) # 记录位移前的流速
            x,y,z= movement_e(g_me,costheta_prev,alpha_prev,beta2,dt,x,y,z)[0:3] # 向上一步变换后的方向移动
            r = np.sqrt(x**2 + y**2)
            
            if (r > R_sh):
                if ESC:
                    gme_jetL[N_time-N_count+1:]= nan
                    r_jetL[N_time-N_count+1:]= nan
                    t_jetL[N_time-N_count+1:]= nan
                    x_jetL[N_time-N_count+1:]= nan
                    y_jetL[N_time-N_count+1:]= nan
                    z_jetL[N_time-N_count+1:]= nan
                    
                    break
                else:
                    x = r0
                    y = 0
                    r = r0
                    
            # 没有出界则更新参数
            t_j += dt
            rg = np.sqrt(g_me**2 - 1) * m_e * c**2 / e / B0
            tau = (rg)**(2-q)*Lam_max**(q-1)*(c*xi)**(-1)
            #print(tau)
            #dt = tau/N_bins
            beta2 = beta_dis(r , GM0, R_sh) # 记录新的beta2, 即位移后的流速
            
            t_jetL[N_time-N_count+1] = t_j
            gme_jetL[N_time-N_count+1] = g_me
            r_jetL[N_time-N_count+1] = r
            x_jetL[N_time-N_count+1] = x
            y_jetL[N_time-N_count+1] = y
            z_jetL[N_time-N_count+1] = z
            
            N_count-=1
            continue
        
        
        # 散射分支，加入随机加速造成的散射方位角
        if Sh:
            g_me = LorentzGamma(beta_ini, beta2, costheta_prev, g_me) # 在beta2处发生散射
            # 直接生成新的移动方向
        
        # 考虑SA时，粒子出射方向只与和磁云的散射有关；不考虑SA时，给粒子指派一个随机的出射方向
        if SA:
            costheta_prev, alpha_prev = Angle_Trans(beta1, beta2, costheta_prev,alpha_prev) # 将方向修正到现有坐标系
            costheta_prev, alpha_prev, g_me = SA_Gamma(costheta_prev, alpha_prev, g_me, ba)
        else:
            poss1 = rng.random()
            poss2 = rng.random()
            costheta_prev = 2*poss1-1
            alpha_prev = 2*pi*poss2
        
        # 同步辐射能损
        if syn:
            g_me -= 1.1e-15 * g_me**2 * B0**2 / m_par / c**2 * dt
            
        beta1 = beta_dis(r, GM0, R_sh) # 记录当前位置流速
        x,y,z = movement_e(g_me,costheta_prev,alpha_prev,beta2,dt,x,y,z)[0:3]
        r = np.sqrt(x**2 + y**2)
        
        # 如果散射后逃逸，则终止循环
        if (r > R_sh):
            if ESC:
                gme_jetL[N_time-N_count+1:]=nan
                r_jetL[N_time-N_count+1:]=nan
                t_jetL[N_time-N_count+1:]=nan
                x_jetL[N_time-N_count+1:] = nan
                y_jetL[N_time-N_count+1:] = nan
                z_jetL[N_time-N_count+1:]= nan
                
                break
            
            else:
                x = r0
                y = 0
                r = r0
                beta1 = beta_ini
                
        # 若散射后不逃逸，则还原粒子位置
        x = r0
        y = 0
        r = r0
        beta1 = beta_ini
        
        # 没有出界则更新参数
        t_j += dt
        rg = np.sqrt(g_me**2 - 1) * m_e * c**2 / e / B0
        tau = (rg)**(2-q)*Lam_max**(q-1)*(c*xi)**(-1)
        #print(tau)
        #dt = tau/N_bins
        #gme_obs = LorentzGamma(beta2, beta_ini, -1, g_me)
        beta2 = beta_dis(r , GM0, R_sh) # 记录新的beta2, 即位移后的流速
        
        t_jetL[N_time-N_count+1] = t_j
        gme_jetL[N_time-N_count+1] = g_me
        r_jetL[N_time-N_count+1] = r
        x_jetL[N_time-N_count+1]=x
        y_jetL[N_time-N_count+1]=y
        z_jetL[N_time-N_count+1]=z
        
        N_count-=1

    return t_jetL, gme_jetL, r_jetL, x_jetL, y_jetL, z_jetL

if __name__=='__main__':
    pool = multiprocessing.Pool(1000)
    numbers = range(N_par) # 0 -- (N-1)
    results = []
with tqdm(total=N_par, desc="Processing particles") as pbar:
    for result in pool.imap_unordered(Single_Par, numbers):
        results.append(result)
        pbar.update(1)
    pool.close()
    pool.join()
    t_jetL = np.array([res[0] for res in results])
    gme_jetL = np.array([res[1] for res in results])
    r_jetL = np.array([res[2] for res in results])
    x_jetL = np.array([res[3] for res in results])
    y_jetL = np.array([res[4] for res in results])
    z_jetL = np.array([res[5] for res in results])
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)   # 删除已有的文件夹
    os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "t_jetL.npy"), t_jetL)
    np.save(os.path.join(output_dir, "gme_jetL.npy"), gme_jetL)
    np.save(os.path.join(output_dir, "r_jetL.npy"), r_jetL)
    np.save(os.path.join(output_dir, "x_jetL.npy"), x_jetL)
    np.save(os.path.join(output_dir, "y_jetL.npy"), y_jetL)
    np.save(os.path.join(output_dir, "z_jetL.npy"), z_jetL)