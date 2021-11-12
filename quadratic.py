from casadi import *
import numpy as np
import itertools


class OCP:  # Quadratic OCP with random Q, c
    newid = itertools.count().__next__

    def __init__(self, nx=1):
        self.id = OCP.newid()
        self.nx = nx
        self.x = SX.sym(f'x{self.id}', self.nx)
        Q = np.random.random((nx, nx)) + nx * np.eye(nx, nx)
        c = np.random.random((nx, 1))
        self.f = 0.5 * self.x.T @ Q @ self.x + c.T @ self.x
        self.qp = {'x': self.x,
                   'f': self.f}
        self.S = qpsol('S', 'qpoases', self.qp)
        print(self.S)

    def solve(self):
        r = self.S(lbg=0)
        xopt = r['x']
        print(f'x_opt : {xopt}')
        return xopt


class Coupling:
    def __init__(self, ocps=None):
        if ocps is None:
            ocps = []
        self.ocps = ocps
        self.coupling_constraints = []
        self.ng = 0
        self.f = None
        self.x = None
        self.qp = None
        self.S = None

    def ocp_by_id(self, id=0):
        for ocp in self.ocps:
            if ocp.id == id:
                return ocp
        raise ValueError('No ocp with specified ID in coupling model.')

    def build_central_problem(self):
        self.f = 0
        xvars = []

        for ocp in self.ocps:
            self.f = self.f + ocp.f
            xvars.append(ocp.x)

        self.x = vcat(xvars)
        self.qp = {'x': self.x,
                   'f': self.f,
                   'g': vcat(self.coupling_constraints)}
        self.S = qpsol('S', 'qpoases', self.qp)

    def solve_central_problem(self):
        r = self.S(lbg=[0]*self.ng, ubg=[0]*self.ng)  # for equality constraints
        xopt = r['x']
        print(f'x_opt : {xopt}')
        return xopt

    def add_coupling(self, new_constraint):
        self.coupling_constraints.append(new_constraint)
        self.ng += 1


if __name__ == '__main__':
    # Create N problems
    N = 3
    nx = 2
    ocps = []
    for i in range(N):
        # Create OCP
        ocp = OCP(nx)
        ocps.append(ocp)

    coupling = Coupling(ocps)
    x_0_1 = coupling.ocp_by_id(0).x[1]
    x_1_0 = coupling.ocp_by_id(1).x[0]
    x_2_1 = coupling.ocp_by_id(2).x[1]
    coupling.add_coupling(x_0_1 - x_1_0)
    coupling.add_coupling(x_2_1 - 2*x_1_0)

    coupling.build_central_problem()
    coupling.solve_central_problem()

