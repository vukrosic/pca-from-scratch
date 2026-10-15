"""PCA from scratch with numpy only.

A direction is a list of 64 numbers of length 1. Pushing a picture onto a
direction gives one number: how far along that direction the picture lies.
The best direction is the one along which the pictures spread the most."""
import warnings

import numpy as np

warnings.filterwarnings("ignore", "divide by zero|overflow|invalid value", RuntimeWarning)  # spurious on Apple matmul


def center(X):
    """Subtract the average picture, so the cloud sits around zero."""
    mean = X.mean(0)
    return X - mean, mean


def spread_along(Xc, d):
    """Variance of the pictures along direction d (d has length 1)."""
    return float(((Xc @ d) ** 2).mean())


def best_direction(Xc, steps=200, seed=0, against=()):
    """Power iteration: start from a random direction, multiply by the cloud's
    spread table, make the length 1 again, repeat. It turns toward the
    direction of most spread. `against` are directions already found: their
    part is removed each round, so the new one is at right angles to them."""
    C = Xc.T @ Xc / len(Xc)                   # 64 x 64 spread table
    d = np.random.default_rng(seed).normal(size=Xc.shape[1])
    trace = []
    for i in range(steps):
        for e in against:
            d = d - (d @ e) * e
        d = C @ d
        for e in against:                     # again: rounding brings them back
            d = d - (d @ e) * e
        n = np.linalg.norm(d)
        if n < 1e-9:                          # no spread left in any new direction
            return d, trace + [0.0]
        d = d / n
        trace.append(spread_along(Xc, d))
    return d, trace


def directions(Xc, k, steps=200):
    """The first k directions, one after the other. Stops early when no
    spread is left (a pixel that is blank in every picture adds none)."""
    found = []
    for i in range(k):
        d, trace = best_direction(Xc, steps, seed=i, against=found)
        if trace[-1] < 1e-9:
            break
        for e in found:                       # remove what is left of earlier ones
            d = d - (d @ e) * e
        found.append(d / np.linalg.norm(d))
    return np.array(found)


def squeeze(Xc, D):
    """Each picture -> k numbers: how far it lies along each direction."""
    return Xc @ D.T


def rebuild(Z, D, mean):
    """k numbers -> a picture: the average plus each direction times its number."""
    return Z @ D + mean
