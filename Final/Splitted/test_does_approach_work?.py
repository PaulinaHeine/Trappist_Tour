from Final.Splitted.Optimize import optimize,optimize_optimal
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

# Dictionary erstellen
#alte ergebnisse von random übernehmen?
#mit normalen einstellungen testen neue perms

test_instances = ["random_g_1","random_1"]
                  #"random_g_2", "random_2","random_g_3", "random_3","random_g_4", "random_4", "random_g_5""random_5"]
solutions = {}
for instance in test_instances:
    solutions[instance] = {"result": {"permutations": "x", "score": "timeout", "runtime": "timeout"}}

# random_100_1

for i in range(len(test_instances)):
    start_time = time.time()
    if i % 2 == 0:
        l = optimize_optimal(run_RNSGA2, 100, run_RNSGA2)
        required = l[6]
        runtime = (time.time() - start_time)
        solutions[test_instances[i]] = {"permutations": required, "score": combine_scores(l[0]), "runtime": runtime}
    else:
        l = optimize(run_RNSGA2, required)
        runtime = (time.time() - start_time)
        solutions[test_instances[i]] = {"permutations": required, "score": combine_scores(l[0]), "runtime": runtime}


#####Code deos not work! Selection of optimize(run_RNSGA2, required)permutations not random (always the same ones)->
#####Because of lack of time code not fixed->Experiment tested manually
##### optimize_optimal(run_RNSGA2, 100, run_RNSGA2)-> optimize(run_RNSGA2, required)