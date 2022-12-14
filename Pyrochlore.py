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


def Solver(x, B_ext):
    S = 1/2
    F = []
    for alpha in [1,2,3,4]:
        sum1, sum2 = 0, 0
        for n_1 in [-1,0,1]:
            for n_2 in [-1,0,1]:
                for n_3 in [-1,0,1]:
                    for beta in [1,2,3,4]:
                        sum1 += (S)*(D(alpha, n_1, n_2, n_3, beta, 1, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
                        sum2 += (S)*(D(alpha, n_1, n_2, n_3, beta, 2, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
        sum1 -= B(1, x[alpha-1], x[alpha+3], B_ext)/2
        sum2 -= B(2, x[alpha-1], x[alpha+3], B_ext)/2
        F.append(sum1)
        F.append(sum2)
    return F


##### Hessian Check ######

def S_Lambda(Lambda, theta_alpha, phi_alpha):
    if(Lambda==3):
        return np.cos(theta_alpha)
    elif(Lambda==1):
        return np.sin(theta_alpha)*np.cos(phi_alpha)
    elif(Lambda==2):
        return np.sin(theta_alpha)*np.sin(phi_alpha)
    else:
        print('Error')


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
    return sum


def mixed_second_derivative(f,i,j,Point):
    h = 1e-6
    a = np.zeros(8)
    b = np.zeros(8)
    c = np.zeros(8)
    d = np.zeros(8)
    for m in range(8):
        a[m] = Point[m]
        b[m] = Point[m]
        c[m] = Point[m]
        d[m] = Point[m]
        if(m==i):
            a[m] = Point[m]-h
            b[m] = Point[m]-h
            c[m] = Point[m]+h
            d[m] = Point[m]+h
        if(m==j):
            a[m] = Point[m]-h
            b[m] = Point[m]+h
            c[m] = Point[m]-h
            d[m] = Point[m]+h
    Derivative = (f(a)+f(d)-f(b)-f(c))/(4*(h**2))
    return Derivative


def Hessian_Check(f, Point):
    H = np.zeros((8,8), dtype=float)
    for i in range(8):
        for j in range(i,8):
            H[i][j] = mixed_second_derivative(f,i,j,Point)
            if(i!=j):
                H[j][i] = H[i][j]
    Eigen_Values = np.linalg.eigvals(H)
    if(Eigen_Values.imag.max()>1e-4): 
        print('Complex Eigen Value Error in Hessian Test')
    else:
        Hessian_EV = np.real_if_close(Eigen_Values, 1e-4)
        #print(Hessian_EV)
        if(Hessian_EV.min() > 0):
            print('Hessian is positive definite')
        else:
            print('Hessian is NOT positive definte')


#guess =  [0.02871551605672237, 0.02871551605672237, 6.254469791122863, 6.254469791122863, 1.542092642554493, 1.5995000110353, 1.5995000110353, 1.542092642554493]

#guess =  [2.18627604, 2.18627604, 0.95531662, 0.95531662, 3.92699082, 0.78539816, 2.35619449, 5.49778714] #J=1,D=0.3,K=0

guess =  [1.547705951521779, 1.593886702068014, 1.593886702068014, 1.547705951521779, 0.023096533230685   ,0.023096533230685   ,6.260088773948902   ,6.260088773948902]

B_ext = [0,0,0]
#root = fsolve(Solver,guess,args=B_ext)
root = guess
Function = Solver(root, B_ext)
print('----------------------------')
print(f'Angles = {root[0],root[1],root[2],root[3], root[4],root[5],root[6],root[7]}')
print('----------------------------')
print('Function value at this Angles =', Function)
print(np.isclose(Solver(root, B_ext), [0, 0, 0 ,0 ,0 ,0 ,0 ,0], atol=1.0e-4))
print('----------------------------')
#Hessian_Check(Classical_Energy_at, root)


# X, B_theta = [], []
# T1, T2, T3, T4 = [], [], [], []
# P1, P2, P3, P4 = [], [], [], []
# B_dir = [1,0,0]
# for i in range(1):
#     B_ext = (i/10)*np.array(B_dir)/np.sqrt(B_dir[0]+B_dir[1]+B_dir[2])
#     B_mag = np.sqrt(np.dot(B_ext, B_ext))
#     X += [B_mag]
#     B_theta += [np.degrees(np.arccos(B_dir[2]/np.sqrt(B_dir[0]+B_dir[1]+B_dir[2])))]
#     convergence = False
#     count = 0
#     while convergence == False and count<=10:
#         root = fsolve(Solver,guess,args=B_ext)
#         if(False in np.isclose(Solver(root, B_ext), [0, 0, 0 ,0 ,0 ,0 ,0 ,0], atol=1.0e-3)):
#             if(count==10):
#                 print('Not_Converging----Not_Converging---Not_Converging')
#             else:
#                 print('Taking Time')
#                 convergence = False
#                 guess = root
#                 count += 1
#         else:
#             convergence = True
#             print(root)
#             T1 += [np.degrees(root[0])]
#             T2 += [np.degrees(root[1])]
#             T3 += [np.degrees(root[2])]
#             T4 += [np.degrees(root[3])]
#             P1 += [np.degrees(root[4])]
#             P2 += [np.degrees(root[5])]
#             P3 += [np.degrees(root[6])]
#             P4 += [np.degrees(root[7])]
#     guess = root  


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
VecStart_x = [0,0,0,2,2,2]
VecStart_y = [0,0,0,2,2,0]
VecStart_z = [0,0,0,0,0,2]
VecEnd_x = [2,0,2,0,2,0]
VecEnd_y = [2,2,0,2,0,2]
VecEnd_z  =[0,2,2,2,2,2]
ax.quiver(-np.cos(root[4])*np.sin(root[0])/2, -np.sin(root[4])*np.sin(root[0])/2, -np.cos(root[0])/2, np.cos(root[4])*np.sin(root[0]), np.sin(root[4])*np.sin(root[0]), np.cos(root[0]), color='c')
ax.quiver(2-np.cos(root[5])*np.sin(root[1])/2, 2-np.sin(root[5])*np.sin(root[1])/2, -np.cos(root[1])/2, np.cos(root[5])*np.sin(root[1]), np.sin(root[5])*np.sin(root[1]), np.cos(root[1]), color='m')
ax.quiver(-np.cos(root[6])*np.sin(root[2])/2, 2-np.sin(root[6])*np.sin(root[2])/2, 2-np.cos(root[2])/2, np.cos(root[6])*np.sin(root[2]), np.sin(root[6])*np.sin(root[2]), np.cos(root[2]), color='y')
ax.quiver(2-np.cos(root[7])*np.sin(root[3])/2, -np.sin(root[7])*np.sin(root[3])/2, 2-np.cos(root[3])/2, np.cos(root[7])*np.sin(root[3]), np.sin(root[7])*np.sin(root[3]), np.cos(root[3]), color='g')
#if(i!= 0): ax.quiver(0,0,0, B_dir[0], B_dir[1], B_dir[2], color='red')
for i in range(6):
    ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]], color='grey')
ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
ax.set_zlim([-1, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#plt.title(f'J={J_int}, D={D_int}, K={K_int}, B={B_mag}[{B_dir[0]}, {B_dir[1]}, {B_dir[2]}]')
plt.show()


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