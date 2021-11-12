from casadi import *
import numpy as np

# Example 2.1
# min_x x_1*x_2 s.t. x_1-x_2 = 0
N = 1
n = 2
m = 1
x = SX.sym('x', n)
f_1 = x[0]*x[1]
A = SX(np.array([1, -1])).T
b = 0

# Solve
# min_y f(y) + lambda.T*A*y + 0.5*rho*norm(A*(y-x))_2^2
z = DM([0, 0])  # Initial guess
lam = 1         # Initial guess
epsilon = 1e-6
err = 1
rho = 0.75      # Divergent for 0.75
k = 0
while err >= epsilon:
    k += 1
    # Solve 'decoupled' NLP
    y = SX.sym('y', n)
    f = y[0]*y[1] + lam@A@y + 0.5*rho*norm_2(A@(y-z))**2
    nlp = {'x': y,
           'f': f}  # No equality constraints
    S = nlpsol('S', 'ipopt', nlp)
    r = S()
    y_opt = r['x']

    # Evaluate error term
    err = float(norm_1(A@y_opt - b))
    if err <= epsilon:
        x_opt = y_opt
        break

    if k >= 100:
        x_opt = y_opt
        break

    # Dual gradient step
    lamplus = lam + rho@A@(y_opt-z)

    # Coupled QP problem
    zplus = SX.sym('z+', n)
    fqp = 0.5*rho*norm_2(A@(y_opt-zplus))**2 - lamplus.T@A@zplus

    qp = {'x': zplus,
          'f': fqp,
          'g': A@zplus - b}
    Sqp = nlpsol('S', 'ipopt', qp)
    rqp = Sqp(ubg=0, lbg=0)  # Equality constraint

    # Update iterates z = zplus, lam = lamplus
    z = r['x']
    lam = lamplus

print(f'x_opt: {y_opt}')
print(f'Iterates: {k}')