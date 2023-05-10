import numpy as np
from matplotlib import pyplot as plt

from Final.Splitted.Sort_and_Filter import sort_points



def plot_all_points(all_sol_t_delta, t_delta):
    '''
    all feasible points and best 100 of pareto front
    '''
    plt.style.use('seaborn-whitegrid')
    all_sol_t_delta = np.array(all_sol_t_delta)
    t_delta = np.array(t_delta)
    plt.figure(figsize=(7, 5))
    plt.scatter(all_sol_t_delta[:, 0], all_sol_t_delta[:, 1], s=40, facecolors='none', edgecolors='steelblue')
    plt.scatter(t_delta[:, 0], t_delta[:, 1], s=40, facecolors='darkgoldenrod', alpha=0.5, edgecolors='darkgoldenrod')
    ref_point_x = 2500
    ref_point_y = 4000
    plt.scatter(ref_point_x, ref_point_y, color="red")
    plt.title("All feasible solutions with pareto front")
    plt.xlim(-5000, 200000)
    plt.ylim(0, 10000)
    plt.xlabel("Delta_V")
    plt.ylabel("Days")
    plt.plot([2500, 2500], [0, 4000], color='r')
    plt.plot([-100000, 2500], [4000, 4000], color='r')
    plt.show()



def plot_front(t_delta):
    '''
    Plot only the best 100 of front
    '''
    plt.style.use('seaborn-whitegrid')
    t_delta = np.array(t_delta)
    plt.figure(figsize=(7, 5))
    plt.scatter(t_delta[:, 0], t_delta[:, 1], s=40, facecolors='darkgoldenrod', alpha=1, edgecolors='darkgoldenrod')
    ref_point_x = 2500
    ref_point_y = 4000
    plt.scatter(ref_point_x, ref_point_y, color="red")
    plt.title("Pareto front")
    plt.xlim(-5000, 20000)
    plt.ylim(0, 10000)
    plt.xlabel("Delta_V")
    plt.ylabel("Days")
    plt.plot([2500, 2500], [0, 4000], color='r')
    plt.plot([-100000, 2500], [4000, 4000], color='r')
    plt.show()



def plot_all_points_behind(all_sol_t_delta, t_delta, conts):
    '''
    All points that give a score (are better then reference point)
    '''
    plt.style.use('seaborn-whitegrid')
    all_sol_t_delta = np.array(all_sol_t_delta)
    t_delta = np.array(t_delta)
    plt.figure(figsize=(7, 5))
    plt.scatter(all_sol_t_delta[:, 0], all_sol_t_delta[:, 1], s=40, facecolors='none', edgecolors='steelblue')
    plt.scatter(t_delta[:, 0], t_delta[:, 1], s=40, facecolors='darkgoldenrod', alpha=0.5, edgecolors='darkgoldenrod')
    t_delta_ref, conts_ref, final_scores_ref = sort_points(t_delta, conts)
    t_delta_ref = np.array(t_delta_ref)
    plt.scatter(t_delta_ref[:, 0], t_delta_ref[:, 1], s=40, facecolors='purple', alpha=1, edgecolors='purple'),
    ref_point_x = 2500
    ref_point_y = 4000
    plt.scatter(ref_point_x, ref_point_y, color="red")
    plt.title("All feasible solutions with pareto front")
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    plt.xlabel("Delta_V")
    plt.ylabel("Days")
    plt.plot([2500, 2500], [0, 4000], color='r')
    plt.plot([-100000, 2500], [4000, 4000], color='r')
    plt.show()



def plot_front_pareto_all_solutions(all_sol_t_delta, pareto_t_delta):
    '''
    nicht nur die hundert besten werden geplottet sondern alle aus der paretofront
    '''
    plt.style.use('seaborn-whitegrid')
    all_sol_t_delta = np.array(all_sol_t_delta)
    pareto_t_delta = np.array(pareto_t_delta)
    plt.figure(figsize=(7, 5))
    plt.scatter(all_sol_t_delta[:, 0], all_sol_t_delta[:, 1], s=40, facecolors='none', edgecolors='steelblue')
    plt.scatter(pareto_t_delta[:, 0], pareto_t_delta[:, 1], s=40, facecolors='darkgoldenrod', alpha=0.5,
                edgecolors='none')
    ref_point_x = 2500
    ref_point_y = 4000
    plt.scatter(ref_point_x, ref_point_y, color="red")
    plt.title("All feasible solutions with pareto front")
    plt.xlim(-5000, 200000)
    plt.ylim(0, 10000)
    plt.xlabel("Delta_V")
    plt.ylabel("Days")
    plt.plot([2500, 2500], [0, 4000], color='r')
    plt.plot([-100000, 2500], [4000, 4000], color='r')
    plt.show()


'''
plt.style.use('seaborn-whitegrid')
all_sol_t_delta = np.array(all_sol_t_delta)
t_delta_ref_2 = np.array(t_delta_ref_2)
plt.figure(figsize=(7, 5))
plt.scatter(all_sol_t_delta[:, 0], all_sol_t_delta[:, 1], s=40, facecolors='none', edgecolors='steelblue')
plt.scatter(t_delta_ref_2[:, 0],t_delta_ref_2[:, 1],s=40, facecolors='purple', alpha=0.5,
            edgecolors='purple')
ref_point_x = 2500
ref_point_y = 4000
plt.scatter(ref_point_x, ref_point_y, color="red")
plt.title("All feasible solutions with pareto front")
plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.xlabel("Delta_V")
plt.ylabel("Days")
plt.plot([2500, 2500], [0, 4000], color='r')
plt.plot([-100000, 2500], [4000, 4000], color='r')
plt.show()
'''
