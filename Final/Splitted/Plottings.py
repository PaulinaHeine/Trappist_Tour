import numpy as np
import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot as plt


def plot_all_points(all_sol_t_delta, t_delta):
    all_sol_t_delta = np.array(all_sol_t_delta)
    t_delta = np.array(t_delta)
    plt.figure(figsize=(7, 5))
    plt.scatter(all_sol_t_delta[:, 0], all_sol_t_delta[:, 1], s=40, facecolors='none', edgecolors='blue')
    plt.scatter(t_delta[:, 0], t_delta[:, 1], s=40, facecolors='green', edgecolors='green')
    ref_point_x = 2500
    ref_point_y = 4000
    plt.scatter(ref_point_x, ref_point_y, color="red")
    plt.title("Pareto font rnsga2")
    plt.xlim(-5000, 200000)
    plt.ylim(0, 10000)
    plt.xlabel("Delta_V")
    plt.ylabel("Days")
    plt.show()



def plot_front(t_delta):
    t_delta = np.array(t_delta)
    plt.figure(figsize=(7, 5))
    plt.scatter(t_delta[:, 0], t_delta[:, 1], s=40, facecolors='none', edgecolors='green')
    ref_point_x = 2500
    ref_point_y = 4000
    plt.scatter(ref_point_x, ref_point_y, color="red")
    plt.title("Pareto font rnsga2")
    plt.xlim(-5000, 200000)
    plt.ylim(0, 10000)
    plt.xlabel("Delta_V")
    plt.ylabel("Days")
    plt.show()

