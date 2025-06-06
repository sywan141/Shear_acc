import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# 方程参数（示例值，需根据实际情况调整）
q = 5/3
A2 = 1e-4
A_stc = 1e-5
A3 = 1
# 计算域设置
gamma_min = 1e6
gamma_max = 1e10
num_points = 1000  # 网格点数

# 转换为对数坐标
x_min = np.log(gamma_min)
x_max = np.log(gamma_max)
x = np.linspace(x_min, x_max, num_points)
dx = x[1] - x[0]

# 初始化系数矩阵（三对角矩阵）
main_diag = np.zeros(num_points-2)
lower_diag = np.zeros(num_points-3)
upper_diag = np.zeros(num_points-3)
b = np.zeros(num_points-2)

# 填充矩阵系数
for i in range(1, num_points-1):
    xi = x[i]
    gamma = np.exp(xi)
    
    # 系数计算
    alpha = 1/dx**2
    beta = (q-2)/(2*gamma*dx) + (2*A2/A_stc)*gamma**(2-q)/(2*dx)
    gamma_coeff = -2*(q-1)/gamma**2 + 2*A2/(A_stc*gamma**(q-1)) - 2/(A3*A_stc*gamma**(2*q-2))
    
    main_diag[i-1] = -2*alpha + gamma_coeff
    if i > 1:
        lower_diag[i-2] = alpha - beta
    if i < num_points-2:
        upper_diag[i-1] = alpha + beta

# 构建稀疏矩阵
diagonals = [main_diag, lower_diag, upper_diag]
A = diags(diagonals, [0, -1, 1], format='csc')

# 边界条件处理（Dirichlet边界）
b[0] = 1e40  # n(gamma_min) = 0
b[-1] = 0  # n(gamma_max) = 0

# 求解线性方程组
n_inner = spsolve(A, b)

# 组合完整解
n = np.zeros(num_points)
n[1:-1] = n_inner

# 转换回伽马空间
gamma = np.exp(x)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.loglog(gamma, n)
print(n)
plt.xlabel('Gamma')
plt.ylabel('n(gamma)')
plt.title('Numerical Solution of the Differential Equation')
plt.grid(True)
plt.savefig('Sol.png')
plt.show()