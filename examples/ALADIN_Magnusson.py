from casadi import *
import numpy as np
import matplotlib.pyplot as plt

def construct_Ci(h, x):
    nh = len(h)
    nx = x.numel()
    Ci = SX.zeros(nh,nx)

    for i in range(nh):
        Ci[i, :] = gradient(h[i], x)

    return Ci

def plot_y_iters(DM_list):
    y_dim = DM_list[0].size(1)
    k_dim = len(DM_list)
    iters = np.zeros([y_dim, k_dim])
    for k in range(0, k_dim):
        iter = DM_list[k]
        for i in range(0, y_dim):
            iters[i, k] = iter[i]

    for i in range(0, y_dim):
        plt.plot(iters[i, :], label=f"y_{i}")

    plt.plot(np.sum(iters*iters, 0), label="f cost")  # x**2 + z**2
    plt.legend()
    plt.show()


# Example 2.1
# min_x x**2 + z**2
# s.t. -2x +z -0.1 = 0
#       0 <= z <= 3
#      -1 <= x <= 2
# -------------------------
#      -z <= 0
#       z - 3 <= 0
#      -x - 1 <= 0
#       x - 2 <= 0
#
#       x[0] : x, x[1] : z

N = 1
n = 2
m = 1
A = SX(np.array([-2, 1])).T
b = 0.1

x0 = x = x_opt = DM([1, 2])  # Initial guess
lam = 1                 # Initial guess
rho = 1                 # Sufficiently large
mu = 1                  # Sufficiently large
# Scaling matrix
Sigma = DM.eye(n)
epsilon = 1e-6
err = 1
k = 0
a1 = a2 = a3 = 1  # Global convergence
y_iters = []

while err >= epsilon:
    k += 1
    # Solve 'decoupled' NLP
    y = SX.sym('y', n)
    fi = y[0]**2 + y[1]**2
    f = fi + lam@A@y + 0.5*rho*(y-x).T@Sigma@(y-x)

    hlist = [-y[1], y[1]-3, -y[0]-1, y[0]-2]
    nlp = {'x': y,
           'f': f,
           'g': vcat(hlist)}
    S = nlpsol('S', 'ipopt', nlp)
    r = S(ubg=[0, 0, 0, 0])
    y_opt = r['x']
    y_iters.append(y_opt)
    kappa = r['lam_g']

    # Evaluate error terms
    err1 = float(norm_1(A@y_opt - b))
    err2 = float(rho*norm_1(Sigma*(y_opt-x)))
    if err1 <= epsilon and err2 <= epsilon:
        x_opt = y_opt
        break

    if k >= 200:
        x_opt = y_opt
        break

    # Construct gi, Ci, Hi
    gi = gradient(fi, y)
    gi = Function('gi', [y], [gi])
    gi = gi(y_opt)
    Ci = construct_Ci(hlist, y)
    Ci = Function('Ci', [y], [Ci])
    Ci = Ci(y_opt)
    Hi = hessian(fi + kappa.T@vcat(hlist), y)[0]
    Hi = Function('Hi', [y], [Hi])
    Hi = Hi(y_opt)

    # Coupled QP with mu >0
    dy = SX.sym('dy', n)
    s = SX.sym('s', m)
    fqp = 0.5*dy.T@Hi@dy+gi.T@dy + lam@s + 0.5*mu*s.T@s
    qp = {'x': vcat([dy, s]),
          'f': fqp,
          'g': vcat([A@(y_opt+dy)-b-s,  # == 0, dual lamQP
                     Ci@dy])}           # == 0, no dual
    Sqp = qpsol('S', 'qpoases', qp)
    rqp = Sqp(ubg=0, lbg=0)  # Eq. constraint
    dy = rqp['x'][0:2]
    lamqp = rqp['lam_g'][0]

    x = x + a1*(y_opt-x) + a2*dy  # == y_opt + dy
    lam = lam + a3*(lamqp - lam)  # == lamqp

print(f'x_opt: {x_opt}')
print(f'Iterates: {k}')
plot_y_iters(y_iters)
