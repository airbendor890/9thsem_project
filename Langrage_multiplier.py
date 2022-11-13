from scipy.optimize import fsolve,  minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S = 1

J_int = 1
D_int = -0.3
K_int = 0.3

B_ext = 0.1*np.array([1,0,0])/np.sqrt(1)

#guess = [2.356194490192345, 2.356194490192345, 0.785398163397448, 0.785398163397448, 1.570796326794897, 4.712388980384690, 4.712388980384690, 1.570796326794897]



fo = open('input_matrix_JDK_hamiltonian_pyrochlore.dat', "r")
ls_eq = fo.read().split('\n')
fo.close()
J_Matrix = np.zeros(3888).reshape(4,3,3,3,4,3,3)
No_data = len(ls_eq)
i = 0
for row in ls_eq:
    i += 1
    data = [float(j) for j in row.split()]
    if(i in range(73)):
        J_Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = J_int*data[-1]/2
    elif(i in range(73,217)):
        J_Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = D_int*data[-1]/2
    else:
        J_Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = K_int*data[-1]
del row, ls_eq, data


def e_trans(m,Lambda,theta,phi):
    #m::local coordinate index
    #lambda::fixed coordinate index
    if(m==1):
        if(Lambda==1):
            return np.cos(theta)*np.cos(phi)
        elif(Lambda==2):
            return np.cos(theta)*np.sin(phi)
        elif(Lambda==3):
            return -np.sin(theta)
    elif(m==2):
        if(Lambda==1):
            return -np.sin(phi)
        elif(Lambda==2):
            return np.cos(phi)
        elif(Lambda==3):
            return 0
    elif(m==3):
        if(Lambda==1):
            return np.sin(theta)*np.cos(phi)
        elif(Lambda==2):
            return np.sin(theta)*np.sin(phi)
        elif(Lambda==3):
            return np.cos(theta)


def J_curly(alpha, n_1, n_2, n_3, beta, Lambda, Mu):
    return J_Matrix[(alpha)-1,(n_1)+1,(n_2)+1,(n_3)+1,(beta)-1,(Lambda)-1,(Mu)-1]


def D(alpha, n_1, n_2, n_3, beta, m, n, theta_alpha, phi_alpha, theta_beta, phi_beta):
    #m,n::local coordinate
    #alpha,beta:sublattice sites
    #i,j :lattice sites
    sum = 0
    for Lambda in [1,2,3]:
        for Mu in [1,2,3]:
            sum += J_curly(alpha, n_1, n_2, n_3, beta, Lambda, Mu)*e_trans(m, Lambda, theta_alpha, phi_alpha)*e_trans(n, Mu, theta_beta, phi_beta)
    return sum


def B(m, theta, phi):
    sum = 0
    for Lambda in range(1,4):
        sum += B_ext[Lambda-1]*e_trans(m, Lambda, theta, phi)
    return sum

def S_Lambda(Lambda, theta_alpha, phi_alpha):
    if(Lambda==3):
        return np.cos(theta_alpha)
    elif(Lambda==1):
        return np.sin(theta_alpha)*np.cos(phi_alpha)
    elif(Lambda==2):
        return np.sin(theta_alpha)*np.sin(phi_alpha)
    else:
        print('Error')


def Linear_Terms(x):
    F = []
    for alpha in [1,2,3,4]:
        sum1, sum2 = 0, 0
        for n_1 in [-1,0,1]:
            for n_2 in [-1,0,1]:
                for n_3 in [-1,0,1]:
                    for beta in [1,2,3,4]:
                        sum1 += (S)*(D(alpha, n_1, n_2, n_3, beta, 1, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
                        sum2 += (S)*(D(alpha, n_1, n_2, n_3, beta, 2, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
        sum1 -= B(1, x[alpha-1], x[alpha+3])/2
        sum2 -= B(2, x[alpha-1], x[alpha+3])/2
        F.append(sum1)
        F.append(sum2)
    return F

def Constrain_1(x, alpha):
    sum = 0
    for n_1 in [-1,0,1]:
        for n_2 in [-1,0,1]:
            for n_3 in [-1,0,1]:
                for beta in [1,2,3,4]:
                    sum += (S)*(D(alpha, n_1, n_2, n_3, beta, 1, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
    sum -= B(1, x[alpha-1], x[alpha+3])/2
    return sum

def Constrain_2(x, alpha):
    sum = 0
    for n_1 in [-1,0,1]:
        for n_2 in [-1,0,1]:
            for n_3 in [-1,0,1]:
                for beta in [1,2,3,4]:
                    sum += (S)*(D(alpha, n_1, n_2, n_3, beta, 2, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
    sum -= B(2, x[alpha-1], x[alpha+3])/2
    return sum


cons = ({'type': 'ineq', 'fun': lambda x:  Constrain_1(x, 1)},
        {'type': 'ineq', 'fun': lambda x:  Constrain_1(x, 2)},
        {'type': 'ineq', 'fun': lambda x:  Constrain_1(x, 3)},
        {'type': 'ineq', 'fun': lambda x:  Constrain_1(x, 4)},
        {'type': 'ineq', 'fun': lambda x:  Constrain_2(x, 1)},
        {'type': 'ineq', 'fun': lambda x:  Constrain_2(x, 2)},
        {'type': 'ineq', 'fun': lambda x:  Constrain_2(x, 3)},
        {'type': 'ineq', 'fun': lambda x:  Constrain_2(x, 4)})


def Classical_Energy_at(Angles):
    sum = 0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                for alpha in range(1,5):
                    for beta in range(1,5):
                        for Lambda in range(1,4):
                            for Mu in range(1,4):
                                sum += J_curly(alpha, n1, n2, n3, beta, Lambda, Mu)*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])*S_Lambda(Mu, Angles[beta-1], Angles[beta+3])
    sum2 =0
    for alpha in range(1,5):
        for Lambda in range(1,4):
            sum2 -= B_ext[Lambda-1]*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])
    return sum/4 + sum2/4


def Energy_minimization(guess):
    res = minimize(Classical_Energy_at, guess, constraints=None, bounds=((0,np.pi), (0,np.pi), (0,np.pi), (0,np.pi), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi)))
    return res

guess2 = np.zeros(8)
#fo = open('Classical_spin_orientations_FM_D_p3.dat', "r")
#fo = open('spin_orientations_fm_D_mp3_h_p1_110.dat', "r")
fo = open('spin_orientations_afm_D_p30_h_p1_110.dat', "r")
#fo = open('spin_orientations_afm_D_mp3_h_p1_100_linear_terms_min.dat', "r")
ls_eq = fo.read().split('\n')
fo.close()
for row in ls_eq:
    data = [float(j) for j in row.split()]
    if(np.round(data[0], 3)== float(K_int)):
        for i in range(8):
            guess2[i] = data[i+1]
        break
del ls_eq, row, data

Energies = []
Angles = []
N = 100
ran = np.random.rand(8*N)
for i in range(N):
    guess = [np.pi*ran[8*i], np.pi*ran[8*i+1], np.pi*ran[8*i+2], np.pi*ran[8*i+3], 2*np.pi*ran[8*i+4], 2*np.pi*ran[8*i+5], 2*np.pi*ran[8*i+6], 2*np.pi*ran[8*i+7]]
    res = Energy_minimization(guess)
    print(res)
    #print('Linear terms:', Linear_Terms(res.x))
    #print('Match:', np.isclose(res.x, guess2, atol=1e-2))
    Energies += [res.fun]
    Angles += [res.x]

Energies = np.array(Energies)
print(Energies.min())
print(Angles[Energies.argmin()])
print('Linear terms:', Linear_Terms(Angles[Energies.argmin()]))
print('Match:', np.isclose(Angles[Energies.argmin()], guess2, atol=1e-2))
print(Classical_Energy_at(guess2))