import itertools

lst = [0, 1, 2, 3, 4, 5, 6]

def all_pairs(lst):
    for p in itertools.permutations(lst):
        i = iter(p)
        yield zip(i,i)