import os
import pickle
import numpy as np

total_acc=[]
total_disp=[]
files = get_files("../../astral-experiments/Temp_experiments")

def get_files(csv_dir):
    csvFilePath =[]
    for path in os.listdir(csv_dir):
        full_path = os.path.join(csv_dir, path)
        csvFilePath.append(full_path)
    NUMBER_FILES = len(csvFilePath)
    return csvFilePath, NUMBER_FILES

for f in files[0]:
    with open(f+"/results/test_log.pkl", 'rb') as f1:
        data = pickle.load(f1)
    acc=[]
    disp=[]

    for i in list(data.values()):
        acc.append(i['client_accs_t'][0])
        disp.append(i['client_disparities_t'][0])
    total_acc.append(acc[-1])
    total_disp.append(disp[-1])


Lmax=[] ; Ldisp=[]
index = np.argpartition([abs(i) for i in total_acc], 3)[:3]
for i in index: 
    Lmax.append(total_acc[i])
    Ldisp.append(total_disp[i])
print(index)
print(Lmax)
print(Ldisp)
