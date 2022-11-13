import Pyrochlore_Lib as lib
import numpy as np
import pandas as pd
import time
#start_time=time.time()

J_int = -1
D_int = 0.3

B_ext = 0.1*np.array([1,0,0])/np.sqrt(1)

VV_Angles = np.zeros(8)
#fo = open('Classical_spin_orientations_FM_D_p3.dat', "r")
fo = open('spin_orientations_fm_D_mp3_h_p1_100.dat', "r")
#fo = open('spin_orientations_afm_D_p30_h_p1_110.dat', "r")
ls_eq = fo.read().split('\n')
fo.close()
for row in ls_eq:
    data = [float(j) for j in row.split()]
    K_int = data[0]
    for i in range(8):
        VV_Angles[i] = data[i+1]
    J_Matrix = lib.Interaction_Matrix(J_int, D_int, K_int)
    root = lib.Ground_State_Finder(B_ext, J_Matrix)
    df1 = pd.DataFrame([{'K_int':{K_int}, 'T1':root[0], 'T2':root[1], 'T3':root[2], 'T4':root[3], 'P1':root[4], 'P2':root[5], 'P3':root[6], 'P4':root[7], 'CE':lib.Classical_Energy_at(root, B_ext, J_Matrix), 'VV_CE':lib.Classical_Energy_at(VV_Angles, B_ext, J_Matrix), 'Max_LT': lib.Linear_Terms(root, B_ext, J_Matrix).max(), 'Max_VV_LT': lib.Linear_Terms(VV_Angles, B_ext, J_Matrix).max()}])
    df1.to_csv('FM_Figures\\J_m1__D_p3.csv', mode='a', index = False, header=False)
    lib.Draw_Band(['G','X','W','G','L','W','U','X','K','G'], B_ext, J_Matrix, J_int, D_int, K_int, root, 'FM_Figures')

#print(time.time()-start_time) 
    
