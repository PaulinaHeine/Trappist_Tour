

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_selection
from pymoo.optimize import minimize
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination

from Final.Splitted.ESA_code import udp



def run_NSGA2(perms, offspring=200, pop_size=50, tol=0.002, n_last=8, n_max_gen=200):

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

                elif len(res) == 5:
                    dv_init, DV, T, lamberts, eq, ineq = udp.fitness(x)
                    dv = dv_init + sum(DV)

                if res == "ERROR: Trajectory infeasible.":
                    if len(res) == 4:
                        dv, T, eq, ineq = udp.fitness(x)
                        dv += 99999999999
                        T += 999999999999999
                    elif len(res) == 5:
                        dv_init, DV, T, lamberts, eq, ineq = udp.fitness(x)
                        dv = dv_init + sum(DV)
                        dv += 99999999999
                        T += 999999999999999

                out["F"] = [dv, T]
                out["G"] = [eq, ineq]

        problem = MyProblem()

        algorithm = NSGA2(
            pop_size= pop_size ,
            n_offsprings= offspring,
            sampling=get_sampling("real_lhs"),
            selection=get_selection('random'),
            crossover=get_crossover("real_sbx", prob=5, eta=15),
            mutation=get_mutation("real_pm", eta=100),
            eliminate_duplicates=True,
            seed=42
        )

        # termination = get_termination("n_gen", 20)
        termination = MultiObjectiveSpaceToleranceTermination(tol= tol,
                                                              n_last= n_last,
                                                              n_max_gen= n_max_gen,
                                                              n_max_evals=None)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=42,
                       save_history=True,
                       verbose=False)

        c = res.X.tolist()

        for x in range(len(c)):
            c[x] += list(perms[n])

        conts += c

        f = res.F.tolist()
        t_delta += f

        print(f"The {n + 1}'th Permutation is done")
        print(f"{len(perms) - (n + 1)} to do ")

    return t_delta, conts

