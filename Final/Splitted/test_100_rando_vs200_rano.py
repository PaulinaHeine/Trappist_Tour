from Final.Splitted.Optimize import optimize
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

test_instances = ["random_100_1", "random_100_2", "random_100_3", "random_100_4", "random_100_5", "random_150_1",
                  "random_150_2",
                  "random_150_3", "random_150_4", "random_150_5"]
solutions = {}
for instance in test_instances:
    solutions[instance] = {"result": {"permutations": "x", "score": "timeout", "runtime": "timeout"}}

# random_100_1
'''
for i in range(len(test_instances)):
    start_time = time.time()
    print(i)
    if i < 5:
        l = optimize(run_RNSGA2, 2)
        runtime = (time.time() - start_time)
        solutions[test_instances[i]] = {"permutations": 100, "score": combine_scores(l[0]), "runtime": runtime}
    else:
        l = optimize(run_RNSGA2, 3)
        runtime = (time.time() - start_time)
        solutions[test_instances[i]] = {"permutations": 150, "score": combine_scores(l[0]), "runtime": runtime}
'''
#random_100_1
start_time = time.time()
l = optimize(run_RNSGA2, 2)
runtime = (time.time() - start_time)
solutions["random_100_1"] = {"permutations": 100, "score": combine_scores(l[0]), "runtime": runtime}

#random_100_2
start_time = time.time()
l1 = optimize(run_RNSGA2, 2)
runtime = (time.time() - start_time)
solutions["random_100_2"] = {"permutations": 100, "score": combine_scores(l1[0]), "runtime": runtime}

#random_100_3
start_time = time.time()
l2 = optimize(run_RNSGA2, 2)
runtime = (time.time() - start_time)
solutions["random_100_3"] = {"permutations": 100, "score": combine_scores(l2[0]), "runtime": runtime}

#random_100_4
start_time = time.time()
l3 = optimize(run_RNSGA2, 2)
runtime = (time.time() - start_time)
solutions["random_100_4"] = {"permutations": 100, "score": combine_scores(l3[0]), "runtime": runtime}

#random_100_5
start_time = time.time()
l4 = optimize(run_RNSGA2, 2)
runtime = (time.time() - start_time)
solutions["random_100_5"] = {"permutations": 100, "score": combine_scores(l4[0]), "runtime": runtime}
