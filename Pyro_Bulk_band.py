from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

J_int = 1
D_int = 0.3
K_int = 0

fo = open('input_matrix_JDK_hamiltonian_pyrochlore.dat', "r")
ls_eq = fo.read().split('\n')
fo.close()
##  alpha=(1,2,3,4), n1,n2,n3=(-1,), beta=(0,1,2,3), mu=(0,1,2), nu=(0,1,2) 
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


def B(m, theta, phi, B_ext):
    sum = 0
    for Lambda in range(1,4):
        sum += B_ext[Lambda-1]*e_trans(m, Lambda, theta, phi)
    return sum


def J_pp(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0+1j)*D(alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) -D(alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

def J_nn(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0+1j)*D(alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) -D(alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))    

def J_pn(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0+1j)*D(alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) +D(alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

def J_np(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0+1j)*D(alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) +D(alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

def J_p3(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(alpha, n_1, n_2, n_3, beta, 1, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 2, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_n3(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(alpha, n_1, n_2, n_3, beta, 1, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 2, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_3p(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(alpha, n_1, n_2, n_3, beta, 3, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 3, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_3n(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(alpha, n_1, n_2, n_3, beta, 3, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(alpha, n_1, n_2, n_3, beta, 3, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_33(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return D(alpha, n_1, n_2, n_3, beta, 3, 3, theta_alpha, phi_alpha, theta_beta, phi_beta)

## 1j = +    and     -1j = -
def J_mn(mu, nu, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    if(mu==1j):
        if(nu==1j):
            return J_pp(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_pn(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==3):
            return J_p3(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    elif(mu==-1j):
        if(nu==1j):
            return J_np(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_nn(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==3):
            return J_n3(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    elif(mu==3):
        if(nu==3):
            return D(alpha, n_1, n_2, n_3, beta, 3, 3, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==1j):
            return J_3p(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_3n(alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    else:
            print('Error')


## Fourier Transform
def J_q(q, mu, nu, alpha, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    sum = 0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                kernal = np.exp((0-1j)*(n1*np.dot(q,np.array([0.5,0.5,0])) + n2*np.dot(q,np.array([0,0.5,0.5])) + n3*np.dot(q,np.array([0.5,0,0.5]))))
                sum += J_mn(mu, nu, alpha, n1, n2, n3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)*kernal
    return sum


def n_q(q):
    # n1 = (q[0]+q[1])/(2*np.pi)
    # n2 = (q[1]+q[2])/(2*np.pi)
    # n3 = (q[0]+q[2])/(2*np.pi)
    # if(abs(np.modf(n1)[0])<1e-3 and abs(np.modf(n2)[0])<1e-3 and abs(np.modf(n3)[0])<1e-3):
    #     return 1
    # else: 
    return 1


def Elements_A_q(q, alpha, beta, Theta_s, Phi_s):
    sum = 0
    if(alpha==beta):
        for gamma in range(1,5):
            sum += J_q(np.array([0,0,0]), 3, 3, alpha, gamma, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[gamma-1], Phi_s[gamma-1]) + J_q(np.array([0,0,0]), 3, 3, gamma, alpha, Theta_s[gamma-1], Phi_s[gamma-1], Theta_s[alpha-1], Phi_s[alpha-1])
    sum = -n_q(q)*sum + 2*n_q(q)*(J_q(-q, -1j, 1j, alpha, beta, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[beta-1], Phi_s[beta-1]) + J_q(q, 1j, -1j, beta, alpha, Theta_s[beta-1], Phi_s[beta-1], Theta_s[alpha-1], Phi_s[alpha-1]))
    return sum

def Elements_B_q(q, alpha, beta, Theta_s, Phi_s):
    return 2*n_q(q)*(J_q(-q, -1j, -1j, alpha, beta, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[beta-1], Phi_s[beta-1]) + J_q(q, -1j, -1j, beta, alpha, Theta_s[beta-1], Phi_s[beta-1], Theta_s[alpha-1], Phi_s[alpha-1]))



Theta_s = [0.9608872073960715, 0.9497240183560501, 2.1807054461937225, 2.1918686352337424]
Phi_s = [0.7736956093192403, 3.9389738214246215, 5.509489697860347, 2.3442114857549665]
B_ext = 0.1*np.array([1,0,0])/np.sqrt(1)
#B_ext = [0.1, 0, 0]

# Theta_s = [1.547705951521779, 1.593886702068014, 1.593886702068014, 1.547705951521779]
# Phi_s = [0  ,0  ,6.260088773948902 ,6.260088773948902]
# B_ext = [0,0,0]

# J =1, D=0.3, K=0, B=[1,0,0] 
# Theta_s = [2.24721644, 2.12756152, 0.89437621, 1.01403113 ]
# Phi_s = [4.07349797, 0.67187216, 2.20968734 ,5.61131314]
# B_ext = [1,0,0]


# J =1 , D =0.3, K =0
# Theta_s = [2.20310149, 2.16964691, 0.93849116, 0.97194575 ]
# Phi_s = [3.96380197, 0.7511103, 2.31938334, 5.53207501]
# B_ext = [0,0,0]

# Theta_s, Phi_s = np.zeros(4), np.zeros(4)
# #B_ext = [0,0,0]
# #fo = open('Classical_spin_orientations_FM_D_p3.dat', "r")
# fo = open('spin_orientations_fm_D_mp3_h_p1_110.dat', "r")
# ls_eq = fo.read().split('\n')
# fo.close()
# for row in ls_eq:
#     data = [float(j) for j in row.split()]
#     if(np.round(data[0], 3)==float(K_int)):
#         for i in range(4):
#             Theta_s[i] = data[i+1]
#             Phi_s[i] = data[5+i]
#         break
# del ls_eq, row, data
# print(Theta_s, Phi_s)

# Theta_s = [1.644230896895485, 1.4973086547196757, 1.5707674703577612, 1.5707725093094291]
# Phi_s = [0.7853638297402888, 0.7853633250895351, 0.7853653270216494, 0.7853710690107436]

def Eigen_Value_finder(q):
    A_q = np.zeros((4,4), dtype=np.csingle)
    B_q = np.zeros((4,4), dtype=np.csingle)
    A_nq= np.zeros((4,4), dtype=np.csingle)

    for alpha in range(4):
        for beta in range(4):
            A_q[alpha, beta] = Elements_A_q(q, alpha+1, beta+1, Theta_s, Phi_s)
            B_q[alpha, beta] = Elements_B_q(q, alpha+1, beta+1, Theta_s, Phi_s)
            A_nq[alpha, beta] = Elements_A_q(-q, alpha+1, beta+1, Theta_s, Phi_s)

    B_q_H = np.conjugate(B_q).T
    A_nq_T = A_nq.T

    if(np.allclose(np.conjugate(A_q).T, A_q) == False):
        print('A_q is NOT Hermitian, Error')

    D_q = np.zeros((8,8), dtype=np.csingle)
    for i in range(4):
        for j in range(4):
            if(i==j):
                D_q[i, j] = A_q[i,j] + B(3, Theta_s[i], Phi_s[i], B_ext)
                D_q[i+4, j+4] = -A_nq_T[i, j] - B(3, Theta_s[i], Phi_s[i], B_ext)
            else:
                D_q[i, j] = A_q[i,j]
                D_q[i+4, j+4] = -A_nq_T[i, j] 
            D_q[i, 4+j] = B_q[i,j]
            D_q[i+4, j] = -B_q_H[i, j]
    Eigen_Values = np.linalg.eigvals(D_q)
    if(Eigen_Values.imag.max()>1e-4): 
        print('Complex Eigen Value Error at q = ', print(q))
        print(Eigen_Values)
    return np.real_if_close(Eigen_Values, 1e-4)

#print(Eigen_Value_finder(np.array([0,0,0])))


def Draw_Band(Path):
    G = np.array([0,0,0])
    X = np.array([2*np.pi,0,0])
    W = np.array([2*np.pi,np.pi,0])
    K = np.array([3*np.pi/2,3*np.pi/2,0])
    L = np.array([np.pi, np.pi, np.pi])
    U = np.array([2*np.pi,np.pi/2, np.pi/2])
    # locals()[str]
    No_points = len(Path) 
    x = []
    y = [[] for i in range(8)]
    x_lebal = []
    q = np.array([0,0,0])
    #Points_in_1_path = 20
    x_l = 0
    for p in range(No_points-1):
        Points_in_1_path = int(np.linalg.norm((locals()[Path[p+1]]-locals()[Path[p]])))*4
        for i in range(Points_in_1_path):
            q = locals()[Path[p]] + (locals()[Path[p+1]]-locals()[Path[p]])*i/Points_in_1_path
            #print(q)
            EV = np.sort(abs(Eigen_Value_finder(q)))
            x += [i+x_l]
            if(i==0): x_lebal += [Path[p]]
            else: x_lebal += [' ']
            for j in range(8):
                y[j] += [EV[j]]
        x_l = x[-1]+1
    EV = np.sort(abs(Eigen_Value_finder(locals()[Path[No_points-1]])))
    x += [x_l]
    x_lebal += [Path[No_points-1]]
    for j in range(8):
        y[j] += [EV[j]]
    
    plt.xticks(x, x_lebal)
    plt.plot(x,y[0])
    plt.plot(x,y[1])
    plt.plot(x,y[2])
    plt.plot(x,y[3])
    plt.plot(x,y[4])
    plt.plot(x,y[5])
    plt.plot(x,y[6])
    plt.plot(x,y[7])
    plt.title(f'J = {J_int}, D = {D_int}, K = {K_int} , B =[{B_ext[0]}, {B_ext[1]}, {B_ext[2]}]')
    plt.show()

Draw_Band(['G','X'])
#Draw_Band(['G','X','W','G','L','W','U','X','K','G'])
#print(Eigen_Value_finder(np.array([np.pi, np.pi, np.pi])))

