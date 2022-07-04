import time

import numpy as np
import itertools
from Final.Splitted.Algorithms.NSGA2_ import run_NSGA2
from Final.Splitted.Algorithms.R_NSGA2_ import run_RNSGA2
from Final.Splitted.ESA_code import combine_scores
from Final.Splitted.Permutations_Sequences import find_perms, permutations, pairwise


def combine_scores_f_perms(points):
    """ Function for aggregating single solutions into one score using hypervolume indicator """
    import pygmo as pg
    ref_point = np.array([2600, 4100])

    # solutions that not dominate the reference point are excluded
    filtered_points = [s for s in points if pg.pareto_dominance(s, ref_point)]

    if len(filtered_points) == 0:
        return 0.0
    else:
        hv = pg.hypervolume(filtered_points)
        return -hv.compute(ref_point)


def find_best_permutations(p, algorithm_p):
    perms = find_perms(r=p)

    print("Start the algorithm with low quality but fast")
    t_delta, conts = algorithm_p(perms, offspring=100, pop_size=70)

    # alle 42 möglichen 2er sequenzen
    permutations_list = list(permutations((0, 1, 2, 3, 4, 5, 6), 2))
    perm_index = list([0] * 42)

    # Für run jeden score berechnen
    final_scores = []
    for t in range(len(t_delta)):
        final_scores.append(combine_scores_f_perms(np.array([t_delta[t]])))

    # für jede lösung den score der 42 möglichkeiten anpassen
    for x in range(len(conts)):
        p_now = list(pairwise(conts[x][-7:]))
        for i in range(len(p_now)):
            for j in range(len(permutations_list)):
                if p_now[i] == permutations_list[j]:
                    if final_scores[x] < 1000:
                        perm_index[j] += 1
                    elif final_scores[x] < 20000:
                        perm_index[j] += 3
                    elif final_scores[x] < 30000:
                        perm_index[j] += 5
                    elif final_scores[x] < 300000:
                        perm_index[j] += 10
                    elif final_scores[x] >= 0:
                        perm_index[j] -= 4
    print(perm_index)
    # beste rausiltern
    best_sequences = []
    new_permutations = []
    for i in range(len(perm_index)):
        if perm_index[i] > 800:
            best_sequences.append(list(permutations_list[i]))
            new_permutations.append(list(permutations_list[i]))

    x = 0
    while x < 5:
        new = []
        for i in range(len(new_permutations)):
            matches = []
            search_planet = new_permutations[i][-1]
            for _ in range(len(best_sequences)):
                if best_sequences[_][0] != search_planet:
                    continue
                else:
                    if best_sequences[_][1] not in new_permutations[i]:
                        matches.append(best_sequences[_])
                        continue
            if len(matches) > 0:
                for m in range(len(matches)):
                    new.append(new_permutations[i] + [matches[m][1]])
            else:
                for e in range(7):
                    if e not in new_permutations[i]:
                        new.append(new_permutations[i] + [e])
                        break

        new_permutations = new
        x += 1
    return new

