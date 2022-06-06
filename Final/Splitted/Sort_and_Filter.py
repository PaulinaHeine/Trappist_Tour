import numpy as np

from Final.Splitted.ESA_code import combine_scores


def proof_dom(t_delta, conts):
    indice = []
    # wird das alles ünernommen??? durchlaufen lassen und werte an indices vergleichen
    all_sol_t_delta = t_delta.copy()
    all_sol_conts = conts.copy()

    # prüfe auf dominanz und sammle alle dominierten punkte
    for x in range(len(t_delta)):
        for y in range(len(t_delta)):
            if t_delta[x][0] > t_delta[y][0]:
                if t_delta[x][1] > t_delta[y][1]:
                    indice.append(x)
        # print(f"{x} done")

    # Duplicate löschen
    indices_2 = list(set(indice))
    # dauert vieeeeel länger, mit reinnehmen in text!!!!!
    # indices_2 = []
    # [indices_2.append(x) for x in indice if x not in indices_2]
    # print(indices_2)

    for x in sorted(indices_2, reverse=True):
        t_delta.pop(x)
        conts.pop(x)
        # print(f"{x} done")

    pareto_t_delta = t_delta.copy()
    pereto_conts = conts.copy()

    return conts, t_delta, all_sol_t_delta, all_sol_conts, pareto_t_delta, pereto_conts

# alle vor dem ref punkt
def sort_points(t_delta, conts):
    # alle punkte hinter dem refpunkt raus
    final_scores_ref = []
    t_delta_ref = t_delta.copy().tolist()
    conts_ref = list(conts.copy())
    for t in range(len(t_delta)):
        final_scores_ref.append(combine_scores(np.array([t_delta[t]])))

    for v in reversed(list(range(len(t_delta)))):
        if final_scores_ref[v] >= 0.0:
            t_delta_ref.pop(v)
            conts_ref.pop(v)
            final_scores_ref.pop(v)


    return t_delta_ref, conts_ref, final_scores_ref


# besten 100 nicht zusammen mit sort points machen
def best_sol(t_delta, conts, bound=100):
    # schlechten punkte hinter refpunkt raus
    final_scores = []

    for t in range(len(t_delta)):
        final_scores.append(combine_scores(np.array([t_delta[t]])))
    all_sol_final_scores = final_scores.copy()

    while len(t_delta) > bound:
        index = final_scores.index(max(final_scores))
        t_delta.pop(index)
        conts.pop(index)
        final_scores.pop(index)
    return t_delta, final_scores, conts, all_sol_final_scores



