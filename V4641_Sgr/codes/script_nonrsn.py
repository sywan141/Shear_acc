import subprocess
import numpy as np

signal = 2
r_values = [1.0]

for r in r_values:
    print(f"\n{'='*30}\n Running R = {r} pc\n{'='*30}")
    
    if(signal==0):
        cmd = ["python", '-u', "/home/wsy/V4641_Sgr/codes/MCMC_nonrsn.py", str(r)]
    elif (signal==1):
        cmd = ["python", '-u', "/home/wsy/V4641_Sgr/codes/MCMC_nonrsn_noHESS.py", str(r)]
    elif (signal==2):
        cmd = ["python", '-u', "/home/wsy/V4641_Sgr/codes/MCMC_SA_4dim.py", str(r)]
    elif (signal==3):
        cmd = ["python", '-u', "/home/wsy/V4641_Sgr/codes/Cooling_free_FP/MCMC_5dim_K1.py", str(r)]
    elif (signal==4):
        cmd = ["python", '-u', "/home/wsy/V4641_Sgr/codes/MCMC_SA_3dim.py", str(r)]
    elif (signal==5):
        cmd = ["python", '-u', "/home/wsy/V4641_Sgr/codes/MCMC_tkin.py", str(r)]

    
    result = subprocess.run(
        cmd,
        text=True,
        check=True  
    )
    
    print(result.stdout)