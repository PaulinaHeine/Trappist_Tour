import itertools
from pymoo.algorithms.moo.nsga2 import NSGA2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from pykep.core import epoch, epoch_from_iso_string, DAY2SEC, EARTH_RADIUS, lambert_problem, propagate_lagrangian, \
    fb_prop, AU, DEG2RAD, ic2par
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler
from pykep.planet import keplerian
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_selection
from pymoo.optimize import minimize
import copy
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
import time
from collections import Counter
import random

'''
# Cavendish constant (m^3/s^2/kg)
G = 6.67430E-11
# Sun_mass (kg)
SM = 1.989E30
# Earth mass (kg)
EM = 5.9722E24
# Mass of the Trappist-1 star
MS = 8.98266512E-2 * SM
# Gravitational parameter of the Trappist system
MU_TRAPPIST = G * MS
# Radius of Trappist-1 [m]
RADIUS_TRAPPIST = 83484000.0
# Maximal allowed distance to Trappist-1 [m]
SAFE_RADIUS_TRAPPIST = 10 * RADIUS_TRAPPIST
# A death penalty for infeasible solutions not directly covered by the constraint handling, i.e. singularities
DV_PENALTY = 1e10
# Starting time of mission
T_START = epoch_from_iso_string("20190302T000000")

# Masses and Keplerian elements a,e,i,W,w,M of the various planets
masses = np.array([1.36004499e+00, 1.29688971e+00, 3.85884170e-01, 6.88627613e-01, 1.03510927e+00, 1.31679654e+00,
                   3.20988718e-01]) * EM
elements = np.array([
    [0.01154035, 0.0158073, 0.02226718, 0.02926548, 0.0385054, 0.04683828, 0.06191385],
    [0.00455247, 0.00107698, 0.00624362, 0.00577129, 0.00861154, 0.00400202, 0.00378361],
    [1.1, 1.3, 0.5, -0.3, 0.01, 1.2, 2.3],
    [238.35940145, 221.02196847, 90.26523049, 164.16001523, 171.42668515, 166.63349296, 282.808809],
    [126.22225948, 80.78036351, 147.44525146, 315.09376191, 179.82855007, 25.31242216, 176.19682451],
    [33.40998384, -89.79506054, -28.38482427, 45.00185452, -145.16404128, 12.59883066, -12.12072197]])
elements = np.transpose(elements)
elements[:, 0] = elements[:, 0] * AU
elements[:, 2:] = elements[:, 2:] * DEG2RAD

safe_radius = np.array([1.016, 1.197, 1.978, 1.020, 1.145, 1.229, 1.875]) * EARTH_RADIUS

# make the planets
planets = []
names = ["b", "c", "d", "e", "f", "g", "h"]
for i in range(7):
    planets.append(keplerian(T_START,
                             elements[i, :6], MU_TRAPPIST, G * masses[i], EARTH_RADIUS, safe_radius[i],
                             "planet_" + names[i]))

# Colors for plotting
pl2c = {'planet_b': 'coral',
        'planet_c': 'seagreen',
        'planet_d': 'purple',
        'planet_e': 'steelblue',
        'planet_f': 'firebrick',
        'planet_g': 'gray',
        'planet_h': 'olive'}


class trappist_tour:
    """
    TOF encoded tour, allowing to constrain time for each leg.
    The decision vector contains of:
    * [u, v, T0] continous part, u and v starting direction, T0 length of first lambert leg
    * for each of the k planetary encounter [beta, rp/rP, eta, T] with beta and rp/rP defining the fly-by, eta the time of the deep space maneuver and T the total duration of the leg. All continuous.
    * [s0, s1, s2, s3, s4, s5, s6] integer part, permutation of [0..6] indicating the visiting sequence

    The initial conditions in starting radius (R_START), velocity magnitude (V_START) and starting epoch (T_START) are
    fixed for this problem (not part of the optimization).
    """

    def __init__(self):
        # there is one leg between each planet plus one additional leg for entering the system
        self.n_legs = len(planets)

        # the travel time between planetary encounters can never be shorter than 5 or longer than 2000 days
        self.tof = [[5.0, 2000.0]] * len(planets)

        # the initial starting conditions of the spacecraft
        self.R_START = 10 * AU
        self.V_START = 1e4
        self.T_START = epoch_from_iso_string("20190302T000000")
        self.common_mu = MU_TRAPPIST

    def get_nobj(self):
        # Our objectives are to minimize DV and total time of flight
        return 2

    def get_nix(self):
        # Integer dimension of the decision vector representing the visiting sequence for the planets
        return 7

    def get_nec(self):
        # sequence of planetary encounters needs to be a permutation
        return 1

    def get_nic(self):
        # checks for keeping a safe distance to the star
        return 1

    def get_bounds(self):
        # I. continuous part (initial leg)
        # we limit v in [0.25, 0.75] which corresponds to a maximum of +/-30 deg Lattitude
        lb = [0.0, 0.25, self.tof[0][0]]
        ub = [1.0, 0.75, self.tof[0][1]]

        # II. continuous part (planetary encounters)
        # encoded as blocks of [beta, rp/rP, eta, T]
        for lower_tof, upper_tof in self.tof[1:]:
            lb += [0, 1.1, 1e-3, lower_tof]
            ub += [2 * np.pi, 100.0, 1.0 - 1e-3, upper_tof]

        # III. integer part
        lb += [0, 0, 0, 0, 0, 0, 0]
        ub += [6, 6, 6, 6, 6, 6, 6]

        return lb, ub

    def _periapsis_passed(self, E0, E, dt, period):
        """ Given two anomaly, the time and the period, is the spacecraft passing the periapsis?
        * E0: starting anomaly
        * E: final anomaly
        * dt: time of flight (in days)
        * period: orbital period (in days)
        """
        if dt > period:
            return True  # over a whole period we pass it for sure
        if E0 > 0:  # spacecraft flying away from body
            return 0 < E < E0
        else:  # flying towards body
            return E > 0 or (dt > period / 2)

    def _check_distance(self, r0, v0, dt, safe_radius=SAFE_RADIUS_TRAPPIST, mu=MU_TRAPPIST):
        """ Computes the periapsis and whether a spacecraft on a certain orbit passed it.
        * r0, v0: initial state of the spacecraft
        * dt: time of flight (in days)
        * safe_radius: minimal allowed distance to central body
        * mu: gravity of central body

        returns (True/False, difference between safe_radius and periapsis)
        """
        # get orbital parameters
        a, e, _, _, _, E0 = ic2par(r0, v0, mu)
        if e <= 1.0:
            rp = a * (1 - e)  # circular orbit
        else:
            rp = -a * (1 - e)  # hyperbolic orbit
        try:
            r, v = propagate_lagrangian(r0, v0, dt * DAY2SEC, mu)
        except RuntimeError:
            # print('ERROR: Trajectory infeasible.')

            # if propagation fails, it typically means that the trajectory passes
            # through the star - consequently we return an extremely high penalty
            return (True, 10e16)

        # get current orbital period [days]
        period = 2 * np.pi * (a ** 3 / mu) ** .5 / DAY2SEC

        # calculate new anomaly
        _, _, _, _, _, E = ic2par(r, v, mu)

        return (self._periapsis_passed(E0, E, dt, period), safe_radius - rp)

    # computation of the objective function
    def fitness(self, x, logging=False, plotting=False, ax=None):
        """ Actual computation of the fitness function
            * x is the chromosome to be evaluated
            * logging toggles detailed output about the encoded trajectory
            * plotting, ax: toggle plotting of the trajectory on the corresponding matplotlib axis
        """
        # split chromosome in continous and integer part
        xc, xi = x[:-7], x[-7:]

        # decode integer part
        seq = [planets[int(i)] for i in xi]

        # check for valid sequences
        eq_constraint = len(set([int(i) for i in xi])) - 7

        # decode continuous part
        u, v = xc[:2]
        T = xc[2::4]
        betas = xc[3::4]
        rps = xc[4::4]
        etas = xc[5::4]

        # starting point on sphere
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1) - np.pi / 2
        rx = self.R_START * np.cos(phi) * np.cos(theta)
        ry = self.R_START * np.cos(phi) * np.sin(theta)
        rz = self.R_START * np.sin(phi)

        # epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs))
        r_P = list([None] * (self.n_legs))
        v_P = list([None] * (self.n_legs))
        lamberts = list([None] * (self.n_legs - 1))
        v_outs = list([None] * (self.n_legs - 1))
        # dv erstmalnur nullen
        DV = list([0.0] * (self.n_legs - 1))

        # violation of distance constraint gets accumulated here
        iq_constraint = -10e16

        # initial starting point
        r_init = [rx, ry, rz]

        for i, planet in enumerate(seq):
            t_P[i] = epoch(self.T_START.mjd2000 + sum(T[0:i + 1]))
            r_P[i], v_P[i] = seq[i].eph(t_P[i])

        # first leg: pure lambert
        lambert_init = lambert_problem(r_init, r_P[0], T[0] * DAY2SEC, self.common_mu, False, 0)

        v_beg_l = lambert_init.get_v1()[0]
        v_end_l = lambert_init.get_v2()[0]

        # the first impulse is discounted by the fact that the spacecraft already starts with a velocity
        DV_init = np.abs(norm(v_beg_l) - self.V_START)

        # checking violation of constraint after first DSM
        close_encounter, d = self._check_distance(r_init, v_beg_l, T[0])
        if close_encounter:
            iq_constraint = max(iq_constraint, d)

        # successive legs
        for i in range(0, self.n_legs - 1):
            # Fly-by
            v_outs[i] = fb_prop(v_end_l, v_P[i], rps[i] * seq[i].radius, betas[i], seq[i].mu_self)

            # updating inequality constraint if necessary
            close_encounter, d = self._check_distance(r_P[i], v_outs[i], etas[i] * T[i + 1])
            if close_encounter:
                iq_constraint = max(iq_constraint, d)

                # s/c propagation before the DSM
            r, v = propagate_lagrangian(r_P[i], v_outs[i], etas[i] * T[i + 1] * DAY2SEC, self.common_mu)
            # Lambert arc to reach next body
            dt = (1 - etas[i]) * T[i + 1] * DAY2SEC
            lamberts[i] = lambert_problem(r, r_P[i + 1], dt, self.common_mu, False, 0)

            v_end_l = lamberts[i].get_v2()[0]
            v_beg_l = lamberts[i].get_v1()[0]

            # DSM occuring at time eta_i*T_i
            if np.isnan(v_beg_l[0]):
                # in rare occassions, the lambert problem is singular or results in unreasonably
                # high velocities. We apply a death penalty to the solution in this case
                print('WARNING: death penalty applied')
                DV[i] = DV_PENALTY
                return (DV_init, DV, T, lamberts, eq_constraint, iq_constraint)

            # updating inequality constraint if necessary
            close_encounter, d = self._check_distance(r, v_beg_l, (1 - etas[i]) * T[i + 1])
            if close_encounter:
                iq_constraint = max(iq_constraint, d)

            DV[i] += norm([a - b for a, b in zip(v_beg_l, v)])

        # pretty printing
        if logging:
            print(f"== 1 : starting point -> {seq[0].name}")
            print(f"Duration: {T[0]:0.6f}d")
            print(f"DV: {DV_init:0.6f}m/s\n")

            for i in range(self.n_legs - 1):
                print(f"== {i + 2} : {seq[i].name} -> {seq[i + 1].name} ==")
                print(f"Duration: {T[i + 1]:0.6f}d")
                print(f"Fly-by epoch: {t_P[i].mjd2000:0.6f} mjd2000")
                print(f"Fly-by radius: {rps[i]:0.6f} planetary radii")
                print(f"DSM after {etas[i] * T[i + 1]:0.6f}d")
                print(f"DSM magnitude: {DV[i]:0.6f}m/s\n")

            print(f"Total Delta V: {sum(DV) + DV_init:0.3f}m/s")
            print(f"Total mission time: {sum(T):0.6f}d ({sum(T) / 365.25:0.3f} years)")

        # plotting
        if plotting:
            ax.scatter(0, 0, 0, color='chocolate')
            for i, planet in enumerate(seq):
                plot_planet(planet, t0=t_P[i], color=pl2c[planet.name], legend=False, units=AU, axes=ax)

            # leg for entering the system
            plot_lambert(lambert_init, sol=0, color='g', legend=False, units=AU, N=10000, axes=ax)

            # intersystem legs
            for i in range(self.n_legs - 1):
                plot_kepler(r_P[i], v_outs[i], etas[i] * T[i + 1] * DAY2SEC, self.common_mu, N=5000, color=(0, 0, 1.0),
                            label=False, units=AU, axes=ax)
            for l in lamberts:
                plot_lambert(l, sol=0, color='r', legend=False, units=AU, N=5000, axes=ax)

            # some settings for better viewing
            zoom = 0.25
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)
            ax.set_zlim(-zoom, zoom)
            ax.margins(x=0)
            ax.view_init(elev=90, azim=90)
            ax.grid(False)
            ax.axis(False)

            # return objectives and constraint violations
        return (DV_init + sum(DV), sum(T), eq_constraint, iq_constraint)

    def pretty(self, x):
        """ Prints out details about the encoded trajectory """
        _ = self.fitness(x, logging=True)

    def plot(self, x, ax=None):
        """ Plots the encoded trajectory in 3d. If no existing matplotlib axis is provided, a new figure is generated """
        if ax is None:

            # changed from:
            # fig = plt.figure()
            # axis = fig.gca(projection='3d')
            # to
            fig = plt.figure(figsize=(10, 8))
            axis = fig.add_subplot(projection='3d')


        else:
            axis = ax

        _ = self.fitness(x, logging=False, plotting=True, ax=axis)
        return axis

    def example(self):
        """ Returns an example solution. """
        return [0.829814333836995, 0.49439092971262544, 1501.6919241751443,
                1.5873030906083387, 1.512021386175137, 0.23024558713584858,
                255.87994897371408, 4.161192563429589, 8.278275659200725, 0.10265621329161562,
                318.6140876998619, 4.546432641418882, 4.078585717496654, 0.33605586368271645,
                374.6287841767241, 0.9461693914845313, 19.13985248269904, 0.4834964750829982,
                343.8204904698013, 2.1935168149524964, 32.48727508444479, 0.40957309270588427,
                325.1537638293067, 4.743610880353654, 49.95062395827792, 0.3668039400426466,
                350.0377039343523, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def combine_scores(points):
    """ Function for aggregating single solutions into one score using hypervolume indicator """
    import pygmo as pg
    ref_point = np.array([2500, 4000])

    # solutions that not dominate the reference point are excluded
    filtered_points = [s for s in points if pg.pareto_dominance(s, ref_point)]

    if len(filtered_points) == 0:
        return 0.0
    else:
        hv = pg.hypervolume(filtered_points)
        return -hv.compute(ref_point)


def combine_scores_sort(points):
    """ Function for aggregating single solutions into one score using hypervolume indicator """
    import pygmo as pg
    ref_point = np.array([5000, 8000])

    # solutions that not dominate the reference point are excluded
    filtered_points = [s for s in points if pg.pareto_dominance(s, ref_point)]

    if len(filtered_points) == 0:
        return 0.0
    else:
        hv = pg.hypervolume(filtered_points)
        return -hv.compute(ref_point)


# Optimize udp
udp = trappist_tour()


############################ Permuatations and sequences ########################

# get all possible zweiersequenzen an Abfolgen

def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n - r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


def find_perms(r=100):
    perms = (list(itertools.permutations([0, 1, 2, 3, 4, 5, 6])))
    # perms = perms[0:100]
    perms = random.sample(perms, r)
    return perms


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


########################## Algorithmns ########################

def run_RNSGA2(perms, offspring=200, pop_size=50,
               ref_points=np.array([[0, 0], [3500, 5000], [2500, 4000], [0, 4000], [2500, 0]]),
               epsilon=0.001, tol=0.002, n_last=8, n_max_gen=200):
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

        algorithm = RNSGA2(
            ref_points=ref_points,
            n_offsprings=offspring,
            pop_size=pop_size,
            epsilon=epsilon,
            normalization='front',
            extreme_points_as_reference_points=False,
            seed=42)

        termination = MultiObjectiveSpaceToleranceTermination(tol=tol,
                                                              n_last=n_last,
                                                              n_max_gen=n_max_gen,
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
            eliminate_duplicates=True
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



############################ Filtering ########################

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

    return conts, t_delta, all_sol_t_delta, all_sol_conts


# alle vor dem ref punkt
def sort_points(t_delta, conts):
    # alle punkte hinter dem refpunkt raus
    final_scores = []
    for t in range(len(t_delta)):
        final_scores.append(combine_scores(np.array([t_delta[t]])))

    for v in reversed(list(range(len(t_delta)))):
        if final_scores[v] >= 0.0:
            t_delta.pop(v)
            conts.pop(v)
            final_scores.pop(v)

    return t_delta, conts, final_scores


# besten 100 nicht zusammen mit sort points machen
def best_sol(t_delta, conts, bound=100):
    # schlechten punkte hinter refpunkt raus
    final_scores = []

    for t in range(len(t_delta)):
        final_scores.append(combine_scores_sort(np.array([t_delta[t]])))
    all_sol_final_scores = final_scores.copy()

    while len(t_delta) > bound:
        index = final_scores.index(max(final_scores))
        t_delta.pop(index)
        conts.pop(index)
        final_scores.pop(index)
    return t_delta, final_scores, conts, all_sol_final_scores


############################ Plots ########################


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
    # plt.savefig("all_sol.png")


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
    # plt.savefig("front.png")


############################ Output & optimize ########################


def find_best_permutations():
    print("Start the run")
    start = time.time()
    print("Select permutations")
    perms = find_perms(10)

    print("Start the algorithm with low quality but fast")
    t_delta, conts = run_NSGA2(perms, offspring=100, pop_size=40,
                               ref_points=np.array([[0, 0], [3500, 5000], [2500, 4000], [0, 4000], [2500, 0]]),
                               epsilon=0.01, tol=0.002, n_last=8, n_max_gen=200)
    print(f"{len(t_delta), len(conts)} if equal everything is fine. We have {len(conts)} solutions found")

    permutations_list = list(permutations((0, 1, 2, 3, 4, 5, 6), 2))
    perm_index = list([0] * 42)

    final_scores = []
    for t in range(len(t_delta)):
        final_scores.append(combine_scores(np.array([t_delta[t]])))

    for x in range(len(conts)):
        p_now = list(pairwise(conts[x][-7:]))
        for i in range(len(p_now)):
            for j in range(len(permutations_list)):
                if p_now[i] == permutations_list[j]:
                    if final_scores[x] < 0:
                        perm_index[j] += 1
                    elif final_scores[x] >= 0:
                        perm_index[j] -= 1

    best_sequences = []
    new_permutations = []
    for i in range(len(perm_index)):
        if perm_index[i] >= -10:
            best_sequences.append(list(permutations_list[i]))
            new_permutations.append(list(permutations_list[i]))

    # new_permutations = [[] for i in range(len(best_sequences))]

    for_perm_now = list([] * 42)
    x = 0
    while x < 5:
        for i in range(len(best_sequences)):
            # best_sequences_i = best_sequences
            search_planet = new_permutations[i][-1]
            for_perm_now.append([])
            for _ in range(len(best_sequences)):
                if best_sequences[_][0] != search_planet and best_sequences[_][1] not in new_permutations[i]:
                    for_perm_now[i].append(best_sequences[_])
                    # if best_sequences[_][0] == search_planet and best_sequences[_][1] not in new_permutations[i]:
                    # new_permutations.append(best_sequences[_])
                    ########new_permutations[i].append(best_sequences[_][1])
                    # index = best_sequences.index(best_sequences[_])
                    # best_sequences.pop(index)
                    # continue
        print(new_permutations)
        x += 1

    stop = time.time()
    time_interval = stop - start

    print(f"The needet time is: {time_interval / 60, round(2)} minutes")


def optimize():
    print("Start the run")
    start = time.time()
    print("Select permutations")
    perms = find_perms(5)

    print("Start the algorithm")
    t_delta, conts = run_RNSGA2(perms)
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


def find_best_perms(conts):
    best_perms_count = []
    for x in range(len(conts)):
        best_perms_count.append(conts[x][1])
    Counter(best_perms_count)
    best_perms = list(set(best_perms_count))
    return best_perms, best_perms_count
'''