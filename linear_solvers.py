"""Classes of linear solvers."""

import sys
import numpy as np
try:
    from mosek import boundkey
    from mosek import Env
    from mosek import objsense
    from mosek import soltype
except ImportError:
    from scipy.optimize import linprog


class LinearSolver:
    def __init__(self):
        pass

    def solve(self, cstr, grad):
        raise NotImplementedError


class AtLeastOneSolver(LinearSolver):
    """ Implements the AtLeastOneSolver. """

    def __init__(self):
        LinearSolver.__init__(self)
        pass

    def solve(self, cstr, grad):
        """Solve the at least one problem."""
        if 'boundkey' in sys.modules:
            x = self._solve_mosek(cstr, grad)
        else:
            x = self._solve_scipy(cstr, grad)
        return np.round(x)

    def _solve_scipy(self, cstr, grad):
        """Solve the at least one problem with scipy."""
        [n, k] = np.shape(grad)

        grad = grad.flatten()
        x = np.zeros(n * k)

        A_ub = None
        b_ub = None
        A_eq = None
        b_eq = None
        bounds = [(0, 1) for _ in range(n*k)]
        for key in cstr.keys():
            if key == 'equal_1':
                for i in cstr[key]:
                    bounds[i] = (1, 1)

            elif key == 'equal_0':
                for i in cstr[key]:
                    bounds[i] = (0, 0)

            elif key == 'row_eq_1':
                A_eq = np.zeros((len(cstr[key]), n*k))
                b_eq = np.ones((len(cstr[key]), 1))
                for i, rows in enumerate(cstr[key]):
                    A_eq[i, rows] = 1

            elif key == 'col_geq_1':
                A_ub = np.zeros((len(cstr[key]), n*k))
                b_ub = -np.ones((len(cstr[key]), 1))
                for i, columns in enumerate(cstr[key]):
                    A_ub[i, columns] = -1

        x = linprog(grad, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='interior-point').x
        return np.reshape(x, [n, k])

    def _solve_mosek(self, cstr, grad):
        """Solve the at least one problem with mosek."""
        [n, k] = np.shape(grad)

        grad = grad.flatten()
        x = np.zeros(n * k)

        with Env() as env:  # Create Environment
            with env.Task(0, 1) as task:  # Create Task
                task.appendvars(n * k)  # 1 variable x

                # Make sure all variables stay in [0,1]
                task.putvarboundlist(
                    np.arange(n * k, dtype=np.int32), [boundkey.ra] * n * k,
                    [0.0] * n * k, [1.0] * n * k)

                # for i, val in enumerate(grad):
                #     task.putcj(i, val)
                task.putclist(np.arange(n * k, dtype=np.int32), grad)

                n_eq_1 = len(cstr['equal_1'])
                # for id_1 in cstr['equal_1']:
                #     task.putvarbound(id_1, boundkey.fx, 1.0, 1.0)
                task.putvarboundlist(cstr['equal_1'], [boundkey.fx] * n_eq_1,
                                     [1.0] * n_eq_1, [1.0] * n_eq_1)

                n_eq_0 = len(cstr['equal_0'])
                # for id_0 in cstr['equal_0']:
                #     task.putvarbound(id_0, boundkey.fx, 0.0, 0.0)
                task.putvarboundlist(cstr['equal_0'], [boundkey.fx] * n_eq_0,
                                     [0.0] * n_eq_0, [0.0] * n_eq_0)

                task.appendcons(len(cstr['row_eq_1']) + len(cstr['col_geq_1']))
                id_con = 0
                for row_cstr_equal_1 in cstr['row_eq_1']:
                    task.putarow(id_con, np.array(row_cstr_equal_1),
                                 np.ones(len(row_cstr_equal_1)))
                    task.putconbound(id_con, boundkey.fx, 1.0, 1.0)
                    id_con += 1

                for col_geq_1 in cstr['col_geq_1']:
                    task.putarow(id_con, col_geq_1, np.ones(len(col_geq_1)))
                    task.putconbound(id_con, boundkey.lo, 1.0, n)
                    id_con += 1

                task.putobjsense(objsense.minimize)  # minimize
                task.optimize()  # Optimize
                task.getxx(soltype.itr, x)

        return np.reshape(x, [n, k])
