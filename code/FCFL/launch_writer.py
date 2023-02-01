import os

tic=0
with open(r'launchTuning.sh','w+') as myBat:
    for i in [0.010, 0.030, 0.050]:
        for j in [1e-4]:
            for p in [1e-4]:
                for k in [0.005, 0.010, 0.020]:
                    for l in [0.005, 0.010, 0.020]:
                        for m in [0.3, 0.8]:
                            for n in [0.3, 0.8]:
                                for o in [5e-3, 5e-4]:  
                                    name=f"test_{i}_{j}_{p}_{k}_{l}_{m}_{n}_{o}"
                                    if name =="test_0.01_0.0001_0.0001_0.02_0.01_0.8_0.8_0.0005":
                                        tic=1
                                    if tic==1:
                                        myBat.write(f'python main.py --target_dir_name {name} --step_size {i} --eps_delta_g {j} --eps_delta_l {p} --factor_delta {k} --lr_delta {l} --delta_l {m} --delta_g {n} --grad_tol {o} --eps_g 0.05 --sensitive_attr sex --max_epoch_stage1 1500 --max_epoch_stage2 2000 --seed 1 --uniform_eps --dataset FCFLdata &&\n')

with open(r'launch.sh','w+') as myBat:
    for i in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070]:
        name=f'test_mb1_{i}'
        myBat.write(f'python main.py --target_dir_name {name} --step_size {i} --eps_g 0.05 --sensitive_attr sex --max_epoch_stage1 2000 --max_epoch_stage2 3000 --seed 1 --uniform_eps --dataset mb1 &&\n')
    for i in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070]:
        name=f'test_mb2_{i}'
        myBat.write(f'python main.py --target_dir_name {name} --step_size {i} --eps_g 0.05 --sensitive_attr sex --max_epoch_stage1 2000 --max_epoch_stage2 3000 --seed 1 --uniform_eps --dataset mb2 &&\n')
    for i in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070]:
        name=f'test_018_4_1_1_{i}'
        myBat.write(f'python main.py --target_dir_name {name} --step_size {i} --eps_g 0.05 --sensitive_attr sex --max_epoch_stage1 2000 --max_epoch_stage2 3000 --seed 1 --uniform_eps --dataset 018_4_1_1 &&\n')



with open(r'launch.sh','rb+') as myBat:
    myBat.seek(-4, os.SEEK_END)
    myBat.truncate()
