
'''
Funktions for permutations and improving permutations
''' 

import random
import itertools


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



def find_perms(r=None):

    #print("Random sample:")
    if r is None:
        r = 10
    ps = list(itertools.permutations([0, 1, 2, 3, 4, 5, 6]))
    perms = random.sample(ps, r)
    #perms = ps[800:r]
    print(perms)
    return perms


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def show_sequence(co):
    # zeige alle sequenzen an
    store = []
    for x in range(len(co)):
        store += [co[x][-7:]]

    return store
