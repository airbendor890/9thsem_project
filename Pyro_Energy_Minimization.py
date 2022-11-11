from scipy.optimize import fsolve, minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# J_int = -1
# D_int = 0.3
# K_int = -0.3

def Interaction_Matrix(J_int, D_int, K_int):
    fo = open('input_matrix_JDK_hamiltonian_pyrochlore.dat', "r")
    ls_eq = fo.read().split('\n')
    fo.close()
    Matrix = np.zeros(3888).reshape(4,3,3,3,4,3,3)
    No_data = len(ls_eq)
    i = 0
    for row in ls_eq:
        i += 1
        data = [float(j) for j in row.split()]
        if(i in range(73)):
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = J_int*data[-1]/2
        elif(i in range(73,217)):
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = D_int*data[-1]/2
        else:
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = K_int*data[-1]
    del row, ls_eq, data
    return Matrix

#J_Matrix = Interaction_Matrix(J_int, D_int, K_int)

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


def J_curly(J_Matrix,alpha, n_1, n_2, n_3, beta, Lambda, Mu):
    return J_Matrix[(alpha)-1,(n_1)+1,(n_2)+1,(n_3)+1,(beta)-1,(Lambda)-1,(Mu)-1]


def D(alpha, n_1, n_2, n_3, beta, m, n, theta_alpha, phi_alpha, theta_beta, phi_beta, J_Matrix):
    #m,n::local coordinate
    #alpha,beta:sublattice sites
    #i,j :lattice sites
    sum = 0
    for Lambda in [1,2,3]:
        for Mu in [1,2,3]:
            sum += J_curly(J_Matrix,alpha, n_1, n_2, n_3, beta, Lambda, Mu)*e_trans(m, Lambda, theta_alpha, phi_alpha)*e_trans(n, Mu, theta_beta, phi_beta)
    return sum


def B(m, theta, phi, B_ext):
    sum = 0
    for Lambda in range(1,4):
        sum += B_ext[Lambda-1]*e_trans(m, Lambda, theta, phi)
    return sum


def Solver(x, B_ext, J_Matrix):
    S = 1
    F = []
    for alpha in [1,2,3,4]:
        sum1, sum2 = 0, 0
        for n_1 in [-1,0,1]:
            for n_2 in [-1,0,1]:
                for n_3 in [-1,0,1]:
                    for beta in [1,2,3,4]:
                        sum1 += (S)*(D(alpha, -n_1, -n_2, -n_3, beta, 1, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3], J_Matrix))
                        sum2 += (S)*(D(alpha, -n_1, -n_2, -n_3, beta, 2, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3], J_Matrix))
        sum1 -= B(1, x[alpha-1], x[alpha+3], B_ext)/2
        sum2 -= B(2, x[alpha-1], x[alpha+3], B_ext)/2
        F.append(sum1)
        F.append(sum2)
    return F


##### Hessian Check ######
def KD(a, b):
    if(a==b):
        return 1
    else:
        return 0

def Theta_Theta_2nd_Diff(beta, gamma, Theta_s, Phi_s, J_Matrix):
    sum = 0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                for alpha in range(1,5):
                    A = -(J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 3, 3) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 3, 3))*(np.cos(Theta_s[gamma-1])*np.cos(Theta_s[alpha-1])*KD(gamma,beta) - np.sin(Theta_s[gamma-1])*np.sin(Theta_s[alpha-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))
                    B = (J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 1, 1) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 1, 1))*(-np.sin(Theta_s[gamma-1])*np.sin(Theta_s[alpha-1])*KD(gamma,beta) + np.cos(Theta_s[gamma-1])*np.cos(Theta_s[alpha-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))*np.cos(Phi_s[gamma-1])*np.cos(Phi_s[alpha-1])
                    C = (J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 2, 2) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 2, 2))*(-np.sin(Theta_s[gamma-1])*np.sin(Theta_s[alpha-1])*KD(gamma,beta) + np.cos(Theta_s[gamma-1])*np.cos(Theta_s[alpha-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))*np.sin(Phi_s[gamma-1])*np.sin(Phi_s[alpha-1])
                    D = (J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 1, 2)*np.cos(Phi_s[gamma-1])*np.sin(Phi_s[alpha-1]) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 1, 2)*np.cos(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1]))*(-np.sin(Theta_s[gamma-1])*np.sin(Theta_s[alpha-1])*KD(gamma,beta) + np.cos(Theta_s[gamma-1])*np.cos(Theta_s[alpha-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))
                    E = (J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 2, 1)*np.sin(Phi_s[gamma-1])*np.cos(Phi_s[alpha-1]) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 2, 1)*np.sin(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1]))*(-np.sin(Theta_s[gamma-1])*np.sin(Theta_s[alpha-1])*KD(gamma,beta) + np.cos(Theta_s[gamma-1])*np.cos(Theta_s[alpha-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))
                    F = (J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 2, 3) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 3, 2))*(-np.sin(Theta_s[gamma-1])*np.cos(Theta_s[alpha-1])*KD(gamma,beta) - np.cos(Theta_s[gamma-1])*np.sin(Theta_s[alpha-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))*np.sin(Phi_s[gamma-1])
                    G = (J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 1, 3) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 3, 1))*(-np.sin(Theta_s[gamma-1])*np.cos(Theta_s[alpha-1])*KD(gamma,beta) - np.cos(Theta_s[gamma-1])*np.sin(Theta_s[alpha-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))*np.cos(Phi_s[gamma-1])
                    H = -(J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 3, 2) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 2, 3))*(np.sin(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta) + np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))*np.sin(Phi_s[alpha-1])
                    I = -(J_curly(J_Matrix,gamma, n1, n2, n3, alpha, 3, 1) + J_curly(J_Matrix,alpha, -n1, -n2, -n3, gamma, 1, 3))*(np.sin(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta) + np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)))*np.cos(Phi_s[alpha-1])
                    sum += A + B + C + D + E  + F + G + H + I
    return sum

def Theta_phi_2nd_Diff(beta,gamma,Theta_s,Phi_s, J_Matrix):
    sum=0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                for alpha in range(1,5):
                    A1=-(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,1,1)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,1,1))*np.cos(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])*(np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.sin(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta)) 
                    A2=(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,2,2)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,2,2))*np.sin(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])*(np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.sin(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta))
                    A3=(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,1,2)*np.cos(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])-J_curly(J_Matrix,gamma,n1,n2,n3,alpha,1,2)*np.sin(Phi_s[gamma-1])*np.sin(Phi_s[alpha-1]))*(np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.sin(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta))
                    A4=(-J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,2,1)*np.sin(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,2,1)*np.cos(Phi_s[gamma-1])*np.cos(Phi_s[alpha-1]))*(np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.sin(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta))
                    A5=(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,3,2)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,2,3))*np.cos(Phi_s[gamma-1])*(-np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.cos(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta))
                    A6=-(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,3,1)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,1,3))*np.sin(Phi_s[gamma-1])*(-np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.cos(Theta_s[alpha-1])*np.cos(Theta_s[gamma-1])*KD(gamma,beta))
                    sum += A1 + A2 + A3 + A4 + A5 + A6
    return sum


def Phi_Phi_2nd_Diff(beta,gamma,Theta_s,Phi_s, J_Matrix):
    sum=0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                for alpha in range(1,5):
                    A1=(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,3,2)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,2,3))*np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(-np.sin(Phi_s[gamma-1])*KD(beta,gamma))
                    A2=(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,3,1)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,1,3))*np.cos(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(np.cos(Phi_s[gamma-1])*KD(beta,gamma))
                    A3=-(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,1,1)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,1,1))*np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(-np.sin(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.cos(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])*KD(gamma,beta)) 
                    A4=(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,2,2)+J_curly(J_Matrix,gamma,n1,n2,n3,alpha,2,2))*np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(np.cos(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)-np.sin(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])*KD(gamma,beta)) 
                    A5=(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,1,2))*np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(-np.sin(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)-np.cos(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])*KD(gamma,beta)) 
                    A6=-(J_curly(J_Matrix,gamma,n1,n2,n3,alpha,1,2))*np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(np.cos(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.sin(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])*KD(gamma,beta)) 
                    A7=-(J_curly(J_Matrix,alpha,-n1,-n2,-n3,gamma,2,1))*np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(np.cos(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)+np.sin(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])*KD(gamma,beta)) 
                    A8=(J_curly(J_Matrix,gamma,n1,n2,n3,alpha,2,1))*np.sin(Theta_s[alpha-1])*np.sin(Theta_s[gamma-1])*(-np.sin(Phi_s[alpha-1])*np.cos(Phi_s[gamma-1])*KD(n1,0)*KD(n2,0)*KD(n3,0)*KD(alpha,beta)-np.cos(Phi_s[alpha-1])*np.sin(Phi_s[gamma-1])*KD(gamma,beta)) 
                    sum += A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8
    return sum

def Hessian_Check(Theta_s, Phi_s, B_ext, J_Matrix):
    S = 1
    H = np.zeros((8,8), dtype=float)
    for alpha in range(4):
        for beta in range(4):
            H[alpha][beta] = (S**2)*Theta_Theta_2nd_Diff(alpha+1, beta+1, Theta_s, Phi_s, J_Matrix) + S*(B_ext[0]*np.sin(Theta_s[beta])*np.cos(Phi_s[beta]) + B_ext[1]*np.sin(Theta_s[beta])*np.sin(Phi_s[beta]) + B_ext[2]*np.cos(Theta_s[beta]))*KD(alpha, beta)
            H[alpha][beta+4] = (S**2)*Theta_phi_2nd_Diff(alpha+1, beta+1, Theta_s, Phi_s, J_Matrix) + S*(B_ext[0]*np.cos(Theta_s[beta])*np.sin(Phi_s[beta]) - B_ext[1]*np.cos(Theta_s[beta])*np.cos(Phi_s[beta]))*KD(alpha, beta)
            H[alpha+4][beta] = (S**2)*Theta_phi_2nd_Diff(beta+1, alpha+1, Theta_s, Phi_s, J_Matrix) + S*(B_ext[0]*np.cos(Theta_s[beta])*np.sin(Phi_s[beta]) - B_ext[1]*np.cos(Theta_s[beta])*np.cos(Phi_s[beta]))*KD(alpha, beta)
            H[alpha+4][beta+4] = (S**2)*Phi_Phi_2nd_Diff(alpha+1, beta+1, Theta_s, Phi_s, J_Matrix) + S*(B_ext[0]*np.sin(Theta_s[beta])*np.cos(Phi_s[beta]) + B_ext[1]*np.sin(Theta_s[beta])*np.sin(Phi_s[beta]))*KD(alpha, beta)
    # for i in range(8):
    #     for j in range(8):
    #         print(np.round(H[i,j],1), end=' ')
    #     print('\n')

    # if(np.allclose(H.T, H)):
    #     print('Hessian is symmetric')
    # Eigen_Values = np.linalg.eigvals(H)
    # if(Eigen_Values.imag.max()>1e-4): 
    #     print('Complex Eigen Value Error in Hessian Test')
    # else:
    #     Hessian_EV = Eigen_Values
    #     print(Hessian_EV)
    #     if(Hessian_EV.min() > 0 or abs(Hessian_EV.min())<=1e-4 ):
    #         print('Hessian is positive definite')
    #     else:
    #         print('Hessian is NOT positive definte')

    try:
        L = np.linalg.cholesky(H)
        print('Hessian is positive definite')
        # for i in range(8):
        #     for j in range(8):
        #         print(np.round(L[i,j],1), end=' ')
        #     print('\n')
    except:
        print('Hessian is NOT positive definite----Error----')

def S_Lambda(Lambda, theta_alpha, phi_alpha):
    S = 1
    if(Lambda==3):
        return S*np.cos(theta_alpha)
    elif(Lambda==1):
        return S*np.sin(theta_alpha)*np.cos(phi_alpha)
    elif(Lambda==2):
        return S*np.sin(theta_alpha)*np.sin(phi_alpha)
    else:
        print('Error')


def Classical_Energy_at(Angles, B_ext, J_Matrix):
    sum = 0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                for alpha in range(1,5):
                    for beta in range(1,5):
                        for Lambda in range(1,4):
                            for Mu in range(1,4):
                                sum += J_curly(J_Matrix,alpha, n1, n2, n3, beta, Lambda, Mu)*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])*S_Lambda(Mu, Angles[beta-1], Angles[beta+3]) 
    sum2 =0
    for alpha in range(1,5):
        for Lambda in range(1,4):
            sum2 -= B_ext[Lambda-1]*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])
    return sum/4 + sum2/4

def Energy_minimization(guess, B_ext, J_Matrix):
    res = minimize(Classical_Energy_at, guess, args=(B_ext, J_Matrix), bounds=((0,np.pi), (0,np.pi), (0,np.pi), (0,np.pi), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi)))
    return res

def Coordinate_Transform(Angles):
    for i in range(4):
        print(f'[x,y,z]_{i+1} = [{np.sin(Angles[i])*np.cos(Angles[i+4])},{np.sin(Angles[i])*np.sin(Angles[i+4])},{np.cos(Angles[i])}]')
    print('\n')

#J =-1, D = 0.3, K = -0.3
#guess =  [1.547705951521779, 1.593886702068014, 1.593886702068014, 1.547705951521779, 0.023096533230685   ,0.023096533230685   ,6.260088773948902   ,6.260088773948902]

# a = 1
# b = 1
# print(D(a,0,0,0,b,1,3,guess[a-1], guess[a+1], guess[b-1], guess[b+1]))

#J = 1 , D = 0.3, K = 0
#guess = [2.18627604, 2.18627604, 0.95531662, 0.95531662, 3.92699082, 0.78539816, 2.35619449, 5.49778714]
#guess = [2.08627604, 2.38627604, 0.85531662, 0.85531662, 3.72699082, 0.77539816, 2.35619449, 5.49778714]

J_int = 1
D_int = 0.3
K_int = 0.3
J_Matrix = Interaction_Matrix(J_int,D_int,K_int)

guess2 = np.zeros(8)
#fo = open('Classical_spin_orientations_FM_D_p3.dat', "r")
#fo = open('spin_orientations_fm_D_mp3_h_p1_110.dat', "r")
fo = open('spin_orientations_afm_D_p30_h_p1_100.dat', "r")
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

# J =1, D = 0.18, K= 0, B= 0
#guess = [0.955316618124509, 0.955316618124509, 2.186276035465284, 2.186276035465284, 0.785398163397448, 3.926990816987241, 5.497787143782138, 2.356194490192345]

# J =1, D = 0.18, K= 0, B= (0.3/np.sqrt(2))*np.array([1, 1, 0])
#guess = [0.9815726067372089, 0.9282017223659544, 2.186102847118882, 2.186102824337781, 0.7853981722203875, 3.926990835473871, 5.554053065524802, 2.299928557535833]
#B_ext = (0.3/np.sqrt(2))*np.array([1, 1, 0])

# J =1, D=0.3, B=0 AIAO state
guess = [0.955316618124509, 0.955316618124509, 2.186276035465284, 2.186276035465284, 0.785398163397448, 3.926990816987241, 5.497787143782138, 2.356194490192345]

# J =1, D= -0.3, B=0
#guess = [2.356194490192345, 2.356194490192345, 0.785398163397448, 0.785398163397448, 1.570796326794897, 4.712388980384690, 4.712388980384690, 1.570796326794897]

#guess = [2, 2, 0.7, 0.7, 1.5, 1.5, 1.5, 1.5]
B_ext = 0.1*np.array([1, 0, 0])/np.sqrt(1)

res = Energy_minimization(guess, B_ext, J_Matrix)
print(res)
Coordinate_Transform(res.x)
Coordinate_Transform(guess2)


Linear_Term_at_guess = Solver(guess2, B_ext, J_Matrix)
print('----------------------------')
print(f'guess = {guess2[0],guess2[1],guess2[2],guess2[3], guess2[4],guess2[5],guess2[6],guess2[7]}')
print('----------------------------') 
print('Function value at this guess =', Linear_Term_at_guess)
print(np.isclose(Solver(guess2, B_ext, J_Matrix), [0, 0, 0 ,0 ,0 ,0 ,0 ,0], atol=1.0e-4))
print('----------------------------')
Hessian_Check([guess2[0],guess2[1],guess2[2],guess2[3]],[guess2[4],guess2[5],guess2[6],guess2[7]], B_ext, J_Matrix)
print('Classical Energy at guess = ', Classical_Energy_at(guess2, B_ext, J_Matrix), '\n')

#root = fsolve(Solver,guess2,args=(B_ext, J_Matrix))
root = res.x
Function = Solver(root, B_ext, J_Matrix)
print('----------------- After Optimization ---------------------')
print(f'Angles = {root[0],root[1],root[2],root[3], root[4],root[5],root[6],root[7]}')
print('----------------------------') 
print('Function value at this Angles =', Function)
print(np.isclose(Solver(root, B_ext, J_Matrix), [0, 0, 0 ,0 ,0 ,0 ,0 ,0], atol=1.0e-4))
print('----------------------------')
Hessian_Check([root[0],root[1],root[2],root[3]],[root[4],root[5],root[6],root[7]], B_ext, J_Matrix)
print('Classical Energy = ', Classical_Energy_at(root, B_ext, J_Matrix))


# X, B_theta = [], []
# T1, T2, T3, T4 = [], [], [], []
# P1, P2, P3, P4 = [], [], [], []
# CE = []
# B_dir = [1,0,0]
# for i in range(11):
#     B_ext = (i/100)*np.array(B_dir)/np.sqrt(B_dir[0]+B_dir[1]+B_dir[2])
#     B_mag = np.sqrt(np.dot(B_ext, B_ext))
#     if(B_mag > 0.1):
#         break
#     X += [B_mag]
#     B_theta += [np.degrees(np.arccos(B_dir[2]/np.sqrt(B_dir[0]+B_dir[1]+B_dir[2])))]
#     convergence = False
#     count = 0
#     while convergence == False and count<=10:
#         root = fsolve(Solver,guess,args=(B_ext, J_Matrix))
#         if(False in np.isclose(Solver(root, B_ext, J_Matrix), [0, 0, 0 ,0 ,0 ,0 ,0 ,0], atol=1.0e-3)):
#             if(count==10):
#                 print('Not_Converging----Not_Converging---Not_Converging')
#             else:
#                 print('Taking Time')
#                 convergence = False
#                 guess = root
#                 count += 1
#         else:
#             convergence = True
#             Hessian_Check([root[0],root[1],root[2],root[3]],[root[4],root[5],root[6],root[7]], B_ext, J_Matrix)
#             print('---------------- B_ext = ', B_ext, '---------------------')
#             print(root)
#             T1 += [np.degrees(root[0])]
#             T2 += [np.degrees(root[1])]
#             T3 += [np.degrees(root[2])]
#             T4 += [np.degrees(root[3])]
#             P1 += [np.degrees(root[4])]
#             P2 += [np.degrees(root[5])]
#             P3 += [np.degrees(root[6])]
#             P4 += [np.degrees(root[7])]
#             CE += [Classical_Energy_at(root, J_Matrix)]
#     guess = root  

# print(np.isclose(guess, guess2, atol = 1e-3))
# print(guess2)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# VecStart_x = [0,0,0,2,2,2]
# VecStart_y = [0,0,0,2,2,0]
# VecStart_z = [0,0,0,0,0,2]
# VecEnd_x = [2,0,2,0,2,0]
# VecEnd_y = [2,2,0,2,0,2]
# VecEnd_z  =[0,2,2,2,2,2]
# ax.quiver(-np.cos(root[4])*np.sin(root[0])/2, -np.sin(root[4])*np.sin(root[0])/2, -np.cos(root[0])/2, np.cos(root[4])*np.sin(root[0]), np.sin(root[4])*np.sin(root[0]), np.cos(root[0]), color='c')
# ax.quiver(2-np.cos(root[5])*np.sin(root[1])/2, 2-np.sin(root[5])*np.sin(root[1])/2, -np.cos(root[1])/2, np.cos(root[5])*np.sin(root[1]), np.sin(root[5])*np.sin(root[1]), np.cos(root[1]), color='m')
# ax.quiver(-np.cos(root[6])*np.sin(root[2])/2, 2-np.sin(root[6])*np.sin(root[2])/2, 2-np.cos(root[2])/2, np.cos(root[6])*np.sin(root[2]), np.sin(root[6])*np.sin(root[2]), np.cos(root[2]), color='y')
# ax.quiver(2-np.cos(root[7])*np.sin(root[3])/2, -np.sin(root[7])*np.sin(root[3])/2, 2-np.cos(root[3])/2, np.cos(root[7])*np.sin(root[3]), np.sin(root[7])*np.sin(root[3]), np.cos(root[3]), color='g')
# if(i!= 0): ax.quiver(0,0,0, B_dir[0], B_dir[1], B_dir[2], color='red')
# for i in range(6):
#     ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]], color='grey')
# ax.set_xlim([-1, 3])
# ax.set_ylim([-1, 3])
# ax.set_zlim([-1, 3])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title(f'J={J_int}, D={D_int}, K={K_int}, B={B_mag}[{B_dir[0]}, {B_dir[1]}, {B_dir[2]}]')
# plt.show()


# plt.scatter(X, T1, label='T1', marker='*')
# plt.scatter(X, T2, label='T2', marker='*')
# plt.scatter(X, T3, label='T3', marker='*')
# plt.scatter(X, T4, label='T4', marker='*')
# plt.scatter(X, P1, label='P1', marker='.')
# plt.scatter(X, P2, label='P2', marker='.')
# plt.scatter(X, P3, label='P3', marker='.')
# plt.scatter(X, P4, label='P4', marker='.')
# plt.plot(X, B_theta)
# plt.xlabel('B')
# plt.ylabel('Angels')
# plt.title(f'J={J_int}, D={D_int}, K={K_int}, B=[{B_dir[0]}, {B_dir[1]}, {B_dir[2]}]')
# plt.legend()
# plt.show()