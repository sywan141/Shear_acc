import subprocess
import numpy as np
import yaml


with open('/home/wsy/Acc_MC/MC_sim/paras.yaml', 'r') as f:
    config = yaml.safe_load(f)
# 模式选择
mode = int(config['mode'])

# 随机行走模式
if (mode==0):
        cmd = ["python", '-u', "/home/wsy/Acc_MC/MC_sim/codes/RW.py"]
# 小角度散射模式
elif (mode==1):
        cmd = ["python", '-u', "/home/wsy/Acc_MC/MC_sim/codes/Rotation_smallAngle.py"]
# 速度变换下的随机行走
elif (mode==2):
        cmd = ["python", '-u', "/home/wsy/Acc_MC/MC_sim/codes/Rotation_RW.py"]
# 拉回中心
elif (mode==3):
        cmd = ["python", '-u', "/home/wsy/Acc_MC/MC_sim/codes/Drag_test/Drag_test.py"] 


result = subprocess.run(
        cmd,
        text=True,
        check=True  
    )