import time

import numpy as np

from Final.Splitted.Algorithms.NSGA2_ import run_NSGA2
from Final.Splitted.Algorithms.R_NSGA2_ import run_RNSGA2
from Final.Splitted.Permutations_Sequences import find_perms
from Final.Splitted.Sort_and_Filter import proof_dom, best_sol
from Final.Splitted.ESA_code import combine_scores
from Final.Splitted.Plottings import plot_all_points, plot_front


def optimize():
    print("Start the run")
    start = time.time()
    print("Select permutations")
    perms = find_perms(10)

    print("Start the algorithm")
    t_delta, conts = run_RNSGA2(perms, offspring = 100)
    print(f"{len(t_delta), len(conts)} if equal everything is fine. We have {len(conts)} solutions found")

    stop = time.time()
    time_interval = stop - start

    print(f"The needed time is: {time_interval / 60} minutes")

    print("Remove the dominated solutions")
    conts, t_delta, all_sol_t_delta, all_sol_conts = proof_dom(t_delta, conts)
    print(f"{len(t_delta), len(conts)} if equal everything is fine. We have {len(conts)} solutions left")

    if combine_scores(np.array(t_delta)) < 0.0:
        print("Score under 0.0 :)")
        if len(t_delta) > 100:
            print("Too many solutions, start filtering ")
            t_delta, final_scores, conts, all_sol_final_scores = best_sol(t_delta, conts)

            stop2 = time.time()
            time_interval = stop2 - start

            print(f"The needed time is: {time_interval / 60} minutes")
            print(f"The score is {combine_scores(t_delta)}")

            return t_delta, conts, final_scores, all_sol_t_delta, all_sol_conts, all_sol_final_scores

    else:
        print("No good solutions found")

    stop2 = time.time()
    time_interval = stop2 - start

    print(f"The needed time is: {time_interval / 60} minutes")
    print(f"The score is {combine_scores(t_delta)}")

    # alle Lösungen weil keine Filterung nach guten Werten notwendig war
    return t_delta, conts, all_sol_t_delta, all_sol_conts


l = optimize()

if len(l) == 4:
    t_delta = l[0]
    conts = l[1]
    all_sol_t_delta = l[2]
    all_sol_conts = l[3]
elif len(l) == 6:
    t_delta = l[0]
    conts = l[1]
    final_scores = l[2]
    all_sol_t_delta = l[3]
    all_sol_conts = l[4]
    all_sol_final_scores = l[5]
