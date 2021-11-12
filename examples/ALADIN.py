from casadi import *
import numpy as np

# Example 2.1
# min_x x_1*x_2 s.t. x_1-x_2 = 0
N = 1
n = 2
m = 1
A = SX(np.array([1, -1])).T
b = 0

x = DM([0, 0])  # Initial guess
lam = 1         # Initial guess
rho = 1         # Sufficiently large
mu = 100          # Sufficiently large
# Scaling matrix
Sigma = 10*DM([[1, 0], [0, 1]])
epsilon = 1e-6
err = 1
k = 0
a1 = a2 = a3 = 1  # Global convergence
while err >= epsilon:
    k += 1
    # Solve 'decoupled' NLP
    y = SX.sym('y', n)
    f = y[0]*y[1] + lam@A@y + 0.5*rho*(y-x).T@Sigma@(y-x)

    nlp = {'x': y,
           'f': f}  # No equality constraint
    S = nlpsol('S', 'ipopt', nlp)
    r = S()
    y_opt = r['x']

    # Evaluate error terms
    err1 = float(norm_1(A@y_opt - b))
    err2 = float(rho*norm_1(Sigma*(y_opt-x)))
    if err1 <= epsilon and err2 <= epsilon:
        x_opt = y_opt
        break

    if k >= 100:
        x_opt = y_opt
        break

    # No ineq constraint so no Ci
    # Gradient of f at y_opt
    g = Function('g', [y], [gradient(f, y)])
    g = g(y_opt)
    # And Hessian
    H = jacobian(gradient(f, y), y)  # No ineq. constraints
    H = Function('H', [y], [H])
    # H = H(y_opt)
    H = rho*A.T@A + n * np.eye(n, n)

    # Coupled QP with mu >0
    dy = SX.sym('dy', n)
    s = SX.sym('s', m)
    fqp = 0.5*dy.T@H@dy+g.T@dy + lam@s + 0.5*mu*s.T@s
    qp = {'x': vcat([dy, s]),
          'f': fqp,
          'g': A@(y_opt+dy)-b-s}  # == 0, dual lamQP
    Sqp = qpsol('S', 'qpoases', qp)
    rqp = Sqp(ubg=0, lbg=0)  # Eq. constraint
    v = rqp['x']
    lamqp = rqp['lam_g']

    x = x + a1*(y_opt-x)  # == y_opt
    lam = lam + a3*(lamqp - lam)  # == lamqp

print(f'x_opt: {x_opt}')
print(f'Iterates: {k}')
