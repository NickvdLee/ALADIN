from casadi import *
import numpy as np

# Example 2.1
# min_x x_1*x_2 s.t. x_1-x_2 = 0

# Problem variables
N = 1
n = 2
m = 1
A = SX(np.array([1, -1])).T
b = 0

# Solver variables
x = x_opt = DM([0, 0])  # Initial guess
lam = 1                 # Initial guess
epsilon = 1e-6
err = 1
rho = 0.75              # Divergent for 0.75
k = 0
while err >= epsilon:
    k += 1
    # Solve 'decoupled' NLP
    y = SX.sym('y', n)
    f = y[0]*y[1] + lam@A@y + 0.5*rho*norm_2(A@(y-x))**2
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
    lamplus = lam + rho*A@(y_opt-x)

    # Coupled QP problem
    xplus = SX.sym('x+', n)
    fqp = 0.5*rho*norm_2(A@(y_opt-xplus))**2 - lamplus.T@A@xplus

    qp = {'x': xplus,
          'f': fqp,
          'g': A@xplus - b}  # Equality constraint
    Sqp = nlpsol('S', 'ipopt', qp)
    rqp = Sqp(ubg=0, lbg=0)  # Equality constraint

    # Update iterates x = xplus, lam = lamplus
    x = r['x']
    lam = lamplus

print(f'x_opt: {x_opt}')
print(f'Iterates: {k}')