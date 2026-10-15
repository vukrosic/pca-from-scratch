"""Checks that the from-scratch PCA agrees with numpy.   python3 tests.py"""
import numpy as np
from sklearn.datasets import load_digits

from pca import center, directions, squeeze, rebuild, spread_along


def main():
    X, _ = load_digits(return_X_y=True)
    Xc, mean = center(X)
    D = directions(Xc, 64)
    assert len(D) == 61, len(D)                          # 3 pixels are blank in every picture
    assert np.allclose(D @ D.T, np.eye(61), atol=1e-8)  # length 1, at right angles
    w = np.linalg.eigvalsh(Xc.T @ Xc / len(Xc))[::-1]
    ours = np.array([spread_along(Xc, d) for d in D])
    assert np.allclose(ours, w[:61], rtol=1e-4)   # 200 rounds: within 0.01 %, (ours[:5], w[:5])
    assert np.allclose(rebuild(squeeze(Xc, D), D, mean), X, atol=1e-6)   # all directions: nothing lost
    total = (Xc ** 2).sum(1).mean()
    assert abs(ours.sum() / total - 1) < 1e-9
    print('all tests pass')


if __name__ == '__main__':
    main()
