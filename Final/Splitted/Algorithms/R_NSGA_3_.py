import numpy as np
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination

from Final.Splitted.ESA_code import udp


##################Value error array beim minimieren ist leer###########################

def run_RNSGA3(perms, offspring=2000,  pop_per_ref_point=50,
               ref_points=np.array([[0, 0], [3500, 5000], [2500, 4000], [0, 4000], [2500, 0]]),
               tol=0.002, n_last=2, n_max_gen=200):
    # Set empty parameters
    t_delta = []

    conts = []

    for n in range(len(perms)):

        class MyProblem(ElementwiseProblem):

            def __init__(self):
                super().__init__(n_var=27,
                                 n_obj=2,
                                 n_constr=2,
                                 xl=np.array(udp.get_bounds()[0][:-7]),
                                 xu=np.array(udp.get_bounds()[1][:-7]))

            def _evaluate(self, x, out, *args, **kwargs):
                x = list(x)
                x += list(perms[n])
                x = np.array(x)
                res = udp.fitness(x)
                if len(res) == 4:
                    dv, T, eq, ineq = udp.fitness(x)

                elif len(res) == 6:
                    dv_init, DV, T, lamberts, eq, ineq = udp.fitness(x)
                    dv = dv_init + sum(DV)

                if res == "ERROR: Trajectory infeasible.":
                    if len(res) == 4:
                        dv, T, eq, ineq = udp.fitness(x)
                        dv += 99999999999
                        T += 999999999999999
                    elif len(res) == 6:
                        dv_init, DV, T, lamberts, eq, ineq = udp.fitness(x)
                        dv = dv_init + sum(DV)
                        dv += 99999999999
                        T += 999999999999999

                out["F"] = [dv, T]
                out["G"] = [eq, ineq]

        problem = MyProblem()

        algorithm = RNSGA3(
            ref_points=ref_points,
            n_offsprings=offspring,
            pop_per_ref_point= pop_per_ref_point,
            seed=42,
            mu=0.1)

        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=n_last,
                                                              n_max_gen=n_max_gen,
                                                              n_max_evals=None)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=42,
                       save_history=True,
                       verbose=True)

        c = res.X.tolist()

        for x in range(len(c)):
            c[x] += list(perms[n])

        conts += c

        f = res.F.tolist()
        t_delta += f

        print(f"The {n + 1}'th Permutation is done")
        print(f"{len(perms) - (n + 1)} to do ")

    return t_delta, conts


