import numpy as np
import time
import math

# 定义常数和参数
NumP = 1000  # 粒子数
NumT = 1000  # 时间步数
NumR = 19    # 半径数

GM0 = 1.1
B0 = 3e-6
e = 4.8e-10
c = 2.998e10
pi = 3.14159
m_e = 9.1e-28
m_p = 1.67e-24
sigTH = 6.65e-25
Lsize = 1e21

# 初始化数组
r0 = np.zeros(NumR)
gme_final = np.zeros((NumR, NumP, NumT))
pos = np.zeros((NumR, NumP, NumT))

# 计算 r0 数组
for i in range(1, 20):  # 1 到 19
    if i <= 10:
        r0[i-1] = (i-1) * Lsize * 0.1
    else:
        r0[i-1] = ((i-10) * 0.01 + 0.9) * Lsize

# 初始条件
gme0 = 100.0 # 初始能量
betae0 = np.sqrt(1 - 1 / gme0**2) # 粒子在初始位置的随动速度
xi = 0.1
rgmax = 2e8 * m_e * c**2 / e / B0
q = 2.0
rg = np.sqrt(gme0**2 - 1) * m_e * c**2 / e / B0 # Lamor半径
mfp = 1 / xi * rg * (rgmax / rg)**(q - 1)
tau0 = mfp / betae0 / c
tesc0 = Lsize**2 / (mfp * betae0 * c / 3)  # 逃逸时标
np_density = 1e-4  # 避免与 numpy 的 np 冲突
va = B0 / np.sqrt(B0**2 + 4 * pi * np_density * m_p * c**2) # Alfven
Dpp = xi * va**2 / (1 - va**2) * gme0**2 * c / rg**(2 - q) / rgmax**(q - 1) # Dpp
tacc0 = gme0**2 / Dpp  # 加速时标

print(f"tacc0: {tacc0}, tesc0: {tesc0}, tau0: {tau0}, mfp: {mfp}")

# gm_dis，喷流各处的Lorentz因子
def gm_dis(r, GM0, rmax):
    r0 = 0
    if abs(r) < r0:
        return GM0
    elif abs(r) > rmax:
        return 1
    else:
        return GM0 - (GM0 - 1) / (rmax - r0) * (abs(r) - r0)

# 定义函数 beta_dis
def beta_dis(r, GM0, rmax):
    r1 = 0
    betamin = 0
    betamax = np.sqrt(1 - 1 / GM0**2)
    if abs(r) < r1:
        return betamax
    elif abs(r) > rmax:
        return 0
    else:
        return betamax + (betamin - betamax) / (rmax - r1) * (abs(r) - r1)

# 开始计时
timei = time.time()

# 主循环
for w in range(1, 2):  # 原代码中 w 从 1 到 1，可改为 range(1, NumR + 1) # 1,2
    print(f"r {r0[w-1]}")
    count_esc = 0

    for n in range(1, NumP + 1):
        print(n)
        tau = tau0
        dt = tau / 1000
        tacc = tacc0
        tesc = tesc0
        gme = gme0
        betae = betae0
        r = r0[w-1]

        # 初始方向
        poss1 = np.random.rand()
        poss2 = np.random.rand()
        costheta = 2 * poss1 - 1
        sintheta = np.sqrt(1 - costheta**2)
        phi = poss2 * 2 * pi
        x = r
        y = 0

        for k in range(1, NumT + 1):
            tacc_bin = 0.001 * tacc0

            while tacc_bin > dt:
                # 更新位置
                x += betae * c * sintheta * np.cos(phi) * dt
                y += betae * c * sintheta * np.sin(phi) * dt
                r_new = np.sqrt(x**2 + y**2)
                dr = r_new - r
                
                # 计算 gm 和 beta
                gm1 = gm_dis(r, GM0, Lsize)
                gm2 = gm_dis(r + dr, GM0, Lsize)
                beta1 = np.sqrt(1 - 1 / gm1**2)
                beta2 = np.sqrt(1 - 1 / gm2**2)
                dbeta = (beta2 - beta1) / (1 - beta2 * beta1)
                #print(dbeta)
                dGM = 1 / np.sqrt(1 - dbeta**2)

                r = abs(r_new)

                # 检查是否逃逸
                if r > Lsize:
                    gme_final[w-1, n-1, k-1] = 0.1
                    pos[w-1, n-1, k-1] = 1e21
                    count_esc += 1
                    break
                
                # 判断是否散射
                poss2 = np.random.rand()
                if poss2 > 1 - np.exp(-dt / tau):
                    tacc_bin -= dt
                    gme = dGM * gme * (1 - betae * dbeta * costheta)
                    gme -= 1.3e-9 * gme**2 * betae**2 * B0**2 * dt  # 同步辐射损失
                    betae = np.sqrt(1 - 1 / gme**2)
                    costheta = (costheta - dbeta) / (1 - costheta * dbeta)
                    sintheta = np.sqrt(1 - costheta**2)
                    tau = mfp / betae / c
                    dt = tau / 1000
                    tesc = Lsize**2 / (mfp * betae * c / 3)
                    continue

                # 散射事件
                while True:
                    rndm3 = np.random.rand()
                    costhetaM = 2 * rndm3 - 1
                    sinthetaM = np.sqrt(1 - costhetaM**2)
                    rndm4 = np.random.rand()
                    phiM = rndm4 * 2 * pi

                    gme1 = dGM * gme * (1 - betae * dbeta * costheta)
                    betae1 = np.sqrt(1 - 1 / gme1**2)
                    costheta1 = (costheta - dbeta) / (1 - costheta * dbeta)
                    sintheta1 = np.sqrt(1 - costheta1**2)
                    phi1 = phi
                    cosinci = costheta1 * costhetaM + sintheta1 * sinthetaM * np.cos(phiM - phi1)
                    gme1 -= 1.3e-9 * gme1**2 * betae1**2 * B0**2 * dt
                    #print(gme1)
                    betae1 = np.sqrt(1 - 1 / gme1**2)
                    poss = np.random.rand()
                    if poss <= (1 - betae1 * va * cosinci) / (1 + betae1 * va):
                        break

                # 计算出射方向
                rndm1 = np.random.rand()
                cosemer = 2 * rndm1 - 1
                sinemer = np.sqrt(1 - cosemer**2)
                rndm2 = np.random.rand()
                phiemer = rndm2 * 2 * pi
                costheta2 = (cosemer + va) / (1 + cosemer * va)
                sintheta2 = np.sqrt(1 - costheta2**2)
                phi2 = phiemer

                cosa = costheta2 * costheta2 + sintheta2 * sintheta2 * np.cos(phi2)
                sina = np.sqrt(1 - cosa**2)
                sinB = sintheta2 * np.sin(phi2) / sina
                cosB = (costheta2 - costheta2 * cosa) / sintheta2 / sina if sina != 0 else 0
                costheta3 = costhetaM * costheta2 - sinthetaM * sintheta2
                sintheta3 = np.sin(np.arccos(costhetaM) + np.arccos(costheta2))
                costheta = costheta3 * cosa + sintheta3 * sina * cosB
                sintheta = np.sqrt(1 - costheta**2)
                cosdphi = (cosa - costheta3 * costheta) / (sintheta3 * sintheta) if sintheta3 * sintheta != 0 else 0
                sindphi = sinB * sina / sintheta if sintheta != 0 else 0

                if cosdphi >= 0 and sindphi >= 0:
                    phi = phiM + np.arcsin(sindphi)
                elif cosdphi <= 0 and sindphi >= 0:
                    phi = phiM + (pi - np.arcsin(sindphi))
                elif cosdphi <= 0 and sindphi <= 0:
                    phi = phiM + (pi - np.arcsin(sindphi))
                else:
                    phi = phiM + (2 * pi + np.arcsin(sindphi))

                if phi > 2 * pi:
                    phi -= 2 * pi

                gme2 = gme1 * (1 / (1 - va**2)) * (1 - betae1 * va * cosinci) * (1 + betae1 * va * cosemer)
                print(dGM * (1 - betae * dbeta * costheta))
                tacc_bin -= dt
                gme = gme2
                rg = np.sqrt(gme**2 - 1) * m_e * c**2 / e / B0
                mfp = 1 / xi * rg * (rgmax / rg)**(q - 1)
                betae = np.sqrt(1 - 1 / gme**2)
                tau = mfp / betae / c
                dt = tau / 1000
                tesc = Lsize**2 / (mfp * betae * c / 3)

            else:
                # 处理剩余时间 tacc_bin
                r_new = np.sqrt((r + betae * c * sintheta * np.cos(phi) * tacc_bin)**2 + 
                                (betae * c * sintheta * np.sin(phi) * tacc_bin)**2)
                dr = r_new - r
                r = abs(r_new)
                beta1 = beta_dis(r, GM0, Lsize)
                beta2 = beta_dis(r + dr, GM0, Lsize)
                dbeta = (beta2 - beta1) / (1 - beta2 * beta1)
                dGM = 1 / np.sqrt(1 - dbeta**2)

                poss1 = np.random.rand()
                if poss1 <= 1 - np.exp(-tacc_bin / tau):
                    if r > Lsize:
                        gme_final[w-1, n-1, k-1] = 0.1
                        pos[w-1, n-1, k-1] = 1e21
                        count_esc += 1
                        break

                    while True:
                        rndm3 = np.random.rand()
                        costhetaM = 2 * rndm3 - 1
                        sinthetaM = np.sqrt(1 - costhetaM**2)
                        rndm4 = np.random.rand()
                        phiM = rndm4 * 2 * pi
                        costheta1 = (costheta - dbeta) / (1 - costheta * dbeta)
                        sintheta1 = np.sqrt(1 - costheta1**2)
                        phi1 = phi
                        cosinci = costheta1 * costhetaM + sintheta1 * sinthetaM * np.cos(phiM - phi1)
                        gme1 = dGM * gme * (1 - betae * dbeta * costheta)
                        betae1 = np.sqrt(1 - 1 / gme1**2)
                        gme1 -= 1.3e-9 * gme1**2 * betae1**2 * B0**2 * tacc_bin
                        betae1 = np.sqrt(1 - 1 / gme1**2)
                        poss = np.random.rand()
                        if poss <= (1 - betae1 * va * cosinci) / (1 + betae1 * va):
                            break

                    rndm1 = np.random.rand()
                    cosemer = 2 * rndm1 - 1
                    sinemer = np.sqrt(1 - cosemer**2)
                    rndm2 = np.random.rand()
                    phiemer = rndm2 * 2 * pi
                    costheta2 = (cosemer + va) / (1 + cosemer * va)
                    sintheta2 = np.sqrt(1 - costheta2**2)
                    phi2 = phiemer

                    cosa = costheta2 * costheta2 + sintheta2 * sintheta2 * np.cos(phi2)
                    sina = np.sqrt(1 - cosa**2)
                    sinB = sintheta2 * np.sin(phi2) / sina if sina != 0 else 0
                    cosB = (costheta2 - costheta2 * cosa) / sintheta2 / sina if sina != 0 else 0
                    costheta3 = costhetaM * costheta2 - sinthetaM * sintheta2
                    sintheta3 = np.sin(np.arccos(costhetaM) + np.arccos(costheta2))
                    costheta = costheta3 * cosa + sintheta3 * sina * cosB
                    sintheta = np.sqrt(1 - costheta**2)
                    cosdphi = (cosa - costheta3 * costheta) / (sintheta3 * sintheta) if sintheta3 * sintheta != 0 else 0
                    sindphi = sinB * sina / sintheta if sintheta != 0 else 0

                    if cosdphi >= 0 and sindphi >= 0:
                        phi = phiM + np.arcsin(sindphi)
                    elif cosdphi <= 0 and sindphi >= 0:
                        phi = phiM + (pi - np.arcsin(sindphi))
                    elif cosdphi <= 0 and sindphi <= 0:
                        phi = phiM + (pi - np.arcsin(sindphi))
                    else:
                        phi = phiM + (2 * pi + np.arcsin(sindphi))

                    gme2 = gme1 * (1 / (1 - va**2)) * (1 - betae1 * va * cosinci) * (1 + betae1 * va * cosemer)
                    tacc_bin = 0
                    gme = gme2
                    rg = np.sqrt(gme**2 - 1) * m_e * c**2 / e / B0
                    mfp = 1 / xi * rg * (rgmax / rg)**(q - 1)
                    betae = np.sqrt(1 - 1 / gme**2)
                    tau = mfp / betae / c
                    dt = tau / 1000
                    tesc = Lsize**2 / (mfp * betae * c / 3)
                else:
                    tacc_bin = 0
                    gme -= 1.3e-9 * gme**2 * betae**2 * B0**2 * tacc_bin
                    betae = np.sqrt(1 - 1 / gme**2)
                    gme = dGM * gme * (1 - betae * dbeta * costheta)
                    betae = np.sqrt(1 - 1 / gme**2)
                    costheta = (costheta - dbeta) / (1 - costheta * dbeta)
                    sintheta = np.sqrt(1 - costheta**2)
                    rg = np.sqrt(gme**2 - 1) * m_e * c**2 / e / B0
                    mfp = 1 / xi * rg * (rgmax / rg)**(q - 1)
                    tau = mfp / betae / c
                    dt = tau / 1000
                    tesc = Lsize**2 / (mfp * betae * c / 3)

            # 记录结果
            if r <= Lsize:  # 只有未逃逸的粒子记录当前步的结果
                gme_final[w-1, n-1, k-1] = gme
                print(gme)
                pos[w-1, n-1, k-1] = r

        else:
            print(f"k {k}")

    print(f"count_esc: {count_esc}")

# 结束计时
timef = time.time()
print(f"Time elapsed: {timef - timei} seconds")

# 保存结果到文件
np.savetxt('gme2_shear52.dat', gme_final.flatten(), fmt='%.18e')
np.savetxt('gme2_shear_pos52.dat', pos.flatten(), fmt='%.18e')