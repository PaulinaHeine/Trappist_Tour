import time

import numpy as np

from Final.Splitted.Algorithms.AGEMOEA import run_AGEMOEA
from Final.Splitted.Algorithms.C_TAEA_ import run_CTAEA
from Final.Splitted.Algorithms.MOEAD_ import run_MOEAD
from Final.Splitted.Algorithms.NSGA2_ import run_NSGA2
from Final.Splitted.Algorithms.R_NSGA2_ import run_RNSGA2
from Final.Splitted.Algorithms.NSGA_3_ import run_NSGA3
from Final.Splitted.Algorithms.R_NSGA_3_ import run_RNSGA3
from Final.Splitted.Algorithms.U_NSGA_3_ import run_UNSGA3
from Final.Splitted.Permutations_Sequences import find_perms, show_sequence
from Final.Splitted.Sort_and_Filter import proof_dom, best_sol
from Final.Splitted.ESA_code import combine_scores
from matplotlib import pyplot as plt
from Final.Splitted.Plottings import plot_all_points, plot_front, plot_all_points_behind
from Final.Splitted.select_improved_perms import find_best_permutations
import random
import itertools


def optimize(algorithm, p):
    print("Start the run")
    start = time.time()
    print("Select permutations")
    perm = find_perms(r=p)
    print(perm)
    print("Start the algorithm")
    t_delta, conts = algorithm(perms=perm, offspring=200, n_max_gen=100)
    print(f"{len(t_delta), len(conts)} if equal everything is fine. We have {len(conts)} solutions found")
    print(f"The score is {combine_scores(t_delta)}")

    stop = time.time()
    time_interval = stop - start

    print(f"The needed time is: {time_interval / 60} minutes")

    print("Remove the dominated solutions")
    conts, t_delta, all_sol_t_delta, all_sol_conts, pareto_t_delta, pereto_conts = proof_dom(t_delta, conts)
    print(f"{len(t_delta), len(conts)} if equal everything is fine. We have {len(conts)} solutions left")
    print(f"The score is {combine_scores(t_delta)}")

    if combine_scores(np.array(t_delta)) < 0.0:
        print("Score under 0.0 :)")
        if len(t_delta) > 999999999999:
            print("Too many solutions, start filtering ")
            t_delta, final_scores, conts, all_sol_final_scores = best_sol(t_delta, conts)

            stop2 = time.time()
            time_interval = stop2 - start

            print(f"The needed time is: {time_interval / 60} minutes")
            print(f"The score is {combine_scores(t_delta)}")

            return t_delta, conts, final_scores, all_sol_t_delta, all_sol_conts, all_sol_final_scores, pareto_t_delta, \
                   pereto_conts
    else:
        print("No good solutions found")

    stop2 = time.time()
    time_interval = stop2 - start

    print(f"The needed time is: {time_interval } seconds")
    print(f"The score is {combine_scores(t_delta)}")
    print(f"Used: {algorithm}")

    # alle Lösungen weil keine Filterung nach guten Werten notwendig war
    return t_delta, conts, all_sol_t_delta, all_sol_conts, pareto_t_delta, pereto_conts


def optimize_optimal(algorithm, p, algorithm_p):
    print("Start the run")
    start = time.time()
    print("Select permutations")
    new = find_best_permutations(p, algorithm_p)
    perms = new
    print(f"{len(perms)}")
    required = len(perms)
    print(perms)

    stop = time.time()
    time_interval = stop - start

    print(f"The needed time to find the permuatations is: {time_interval / 60} minutes")

    print("Start the algorithm")
    t_delta, conts = algorithm(perms, offspring=200, n_max_gen=150)
    print(f"{len(t_delta), len(conts)} if equal everything is fine. We have {len(conts)} solutions found")
    print(f"The score is {combine_scores(t_delta)}")

    stop = time.time()
    time_interval = stop - start

    print(f"The needed time is: {time_interval / 60} minutes")

    print("Remove the dominated solutions")
    conts, t_delta, all_sol_t_delta, all_sol_conts, pareto_t_delta, pereto_conts = proof_dom(t_delta, conts)
    print(f"{len(t_delta), len(conts)} if equal everything is fine. We have {len(conts)} solutions left")
    print(f"The score is {combine_scores(t_delta)}")

    if combine_scores(np.array(t_delta)) < 0.0:
        print("Score under 0.0 :)")
        if len(t_delta) > 9999999999999:
            print("Too many solutions, start filtering ")
            t_delta, final_scores, conts, all_sol_final_scores = best_sol(t_delta, conts)

            stop2 = time.time()
            time_interval = stop2 - start

            print(f"The needed time is: {time_interval / 60} minutes")
            print(f"The score is {combine_scores(t_delta)}")

            return t_delta, conts, final_scores, all_sol_t_delta, all_sol_conts, all_sol_final_scores, pareto_t_delta, \
                   pereto_conts
    else:
        print("No good solutions found")

    stop2 = time.time()
    time_interval = stop2 - start

    print(f"The needed time is: {time_interval / 60} minutes")
    print(f"The score is {combine_scores(t_delta)}")
    print(f"Used: {algorithm}")

    # alle Lösungen weil keine Filterung nach guten Werten notwendig war
    return t_delta, conts, all_sol_t_delta, all_sol_conts, pareto_t_delta, pereto_conts, required


#l = optimize(run_RNSGA2, 200)
'''
# r = optimize_optimal(run_RNSGA2, 30, run_RNSGA2)


if len(l) == 6:
    t_delta = l[0]
    conts = l[1]
    all_sol_t_delta = l[2]
    all_sol_conts = l[3]
    pareto_t_delta = l[4]
    pereto_conts = l[5]

elif len(l) == 8:
    t_delta = l[0]
    conts = l[1]
    final_scores = l[2]
    all_sol_t_delta = l[3]
    all_sol_conts = l[4]
    all_sol_final_scores = l[5]
    pareto_t_delta = l[6]
    pareto_conts = l[7]

if len(r) == 7:
    t_delta = r[0]
    conts = r[1]
    all_sol_t_delta = r[2]
    all_sol_conts = r[3]
    pareto_t_delta = r[4]
    pereto_conts = r[5]
    reqired=r[6]

elif len(r) == 8:
    t_delta = r[0]
    conts = r[1]
    final_scores = r[2]
    all_sol_t_delta = r[3]
    all_sol_conts = r[4]
    all_sol_final_scores = r[5]
    pareto_t_delta = r[6]
    pareto_conts = r[7]
'''
# plot_all_points(all_sol_t_delta, t_delta)
# plot_front(t_delta)
# show_sequence(conts)

# plot_all_points_behind(all_sol_t_delta, t_delta, conts)
