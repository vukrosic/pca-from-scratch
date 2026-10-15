"""Exercise: find the best direction yourself.

Fill in best_direction below, then run:   python3 exercise.py
It prints how much of the spread your direction keeps and compares it
with numpy's answer. The answer is in solution.py."""
import warnings

import numpy as np
from sklearn.datasets import load_digits

warnings.filterwarnings("ignore", "divide by zero|overflow|invalid value", RuntimeWarning)  # spurious on Apple matmul


def best_direction(Xc, steps=200):
    """Xc: the pictures with the average picture subtracted (1797 x 64).
    Return a direction of length 1 along which the pictures spread most."""
    d = np.random.default_rng(0).normal(size=64)   # a random direction
    # your part starts here
    # repeat `steps` times:
    #   1. each picture's number along d:   Xc @ d
    #   2. every picture pulls as hard as its number; add the pulls:   Xc.T @ numbers
    #   3. divide d by its length, so its length is 1 again
    # and ends here
    return d / np.linalg.norm(d)


if __name__ == '__main__':
    X, _ = load_digits(return_X_y=True)
    Xc = X - X.mean(0)
    d = best_direction(Xc)
    total = (Xc ** 2).sum(1).mean()
    spread = ((Xc @ d) ** 2).mean()
    best = np.linalg.eigvalsh(Xc.T @ Xc / len(Xc))[-1]
    print(f'your spread: {spread:.1f} (keeps {spread / total:.1%}); the best possible: {best:.1f} ({best / total:.1%})')
    print('right!' if abs(spread - best) < 1e-4 else 'not yet: the spread should match')
