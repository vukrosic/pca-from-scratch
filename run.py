"""Every number in the film. Writes files/results.json. About a minute."""
import json
import time

import numpy as np
from sklearn.datasets import load_digits

from pca import center, spread_along, best_direction, directions, squeeze, rebuild

KS = (1, 2, 5, 10, 20)


def nearest_neighbour_right(train, ytr, test, yte):
    """Guess each test picture's digit by the closest training picture."""
    d = ((test[:, None, :] - train[None, :, :]) ** 2).sum(-1)
    return float((ytr[d.argmin(1)] == yte).mean())


def main():
    t0 = time.time()
    X, y = load_digits(return_X_y=True)
    out = {'pictures': len(X), 'numbers_per_picture': X.shape[1], 'max_value': float(X.max())}
    Xc, mean = center(X)
    total = float((Xc ** 2).sum(1).mean())          # all the spread there is
    out['total_spread'] = total

    # 1. one picture as 64 numbers
    out['one'] = {'digit': int(y[0]), 'numbers': X[0].astype(int).tolist()}

    # 1b. only two pixels: one just above the other (pixels 13 and 21), as a dot cloud
    T = X[:, [13, 21]]
    Tc, tm = center(T)
    ang = np.radians(np.arange(0, 180, 5))
    sweep = [spread_along(Tc, np.array([np.cos(t), np.sin(t)])) for t in ang]
    d30 = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    dt, _ = best_direction(Tc)
    out['two'] = {'pixels': [13, 21], 'points': T.astype(int).tolist(), 'mean': tm.round(2).tolist(),
                  'example': {'values': T[0].tolist(), 'diff': Tc[0].round(2).tolist(), 'weights': d30.round(3).tolist(),
                              'number': float(Tc[0] @ d30)},
                  'sweep_deg': list(range(0, 180, 5)), 'sweep': [float(v) for v in sweep],
                  'best_deg': float(np.degrees(np.arctan2(dt[1], dt[0])) % 180), 'best': spread_along(Tc, dt),
                  'total': float((Tc ** 2).sum(1).mean()), 'corr': float(np.corrcoef(T.T)[0, 1])}

    # 2. the best line: random directions vs power iteration
    rng = np.random.default_rng(1)
    rnd = [spread_along(Xc, v / np.linalg.norm(v)) for v in rng.normal(size=(1000, 64))]
    d1, trace = best_direction(Xc)
    out['line'] = {'random_mean': float(np.mean(rnd)), 'random_best': float(np.max(rnd)),
                   'trace': [trace[i] for i in (0, 1, 2, 3, 5, 10, 20, 50, 199)], 'best': trace[-1],
                   'share_random_mean': float(np.mean(rnd)) / total, 'share_best': trace[-1] / total}

    # 3. check against numpy's own answer
    w, V = np.linalg.eigh(Xc.T @ Xc / len(Xc))
    w, V = w[::-1], V[:, ::-1]
    D = directions(Xc, 64)
    out['check'] = {'largest_eigenvalue': float(w[0]), 'ours': trace[-1],
                    'angle_deg_first5': [float(np.degrees(np.arccos(min(1, abs(D[i] @ V[:, i]))))) for i in range(5)]}

    # 4. spread kept by number of directions
    share = np.cumsum([spread_along(Xc, d) for d in D]) / total
    out['kept'] = {k: float(share[k - 1]) for k in range(1, len(D) + 1)}
    out['directions_found'] = len(D)
    out['blank_pixels'] = int((X.std(0) == 0).sum())
    out['need'] = {str(p): int(np.searchsorted(share, p) + 1) for p in (0.5, 0.8, 0.9, 0.95, 0.99)}

    # 5. rebuild pictures from k directions
    show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]               # the first picture of each digit
    first = [int(np.where(y == dg)[0][0]) for dg in show]
    out['rebuild'] = {'ids': first, 'original': [X[i].round(1).tolist() for i in first]}
    for k in KS:
        R = rebuild(squeeze(Xc, D[:k]), D[:k], mean)
        out['rebuild'][k] = {'pictures': [R[i].round(1).tolist() for i in first],
                             'error': float(((R - X) ** 2).sum(1).mean()), 'kept': float(share[k - 1])}
    out['rebuild']['error_none'] = total
    out['directions_as_pictures'] = [D[i].round(3).tolist() for i in range(6)]

    # 6. the map: every picture as 2 numbers
    Z = squeeze(Xc, D[:2])
    out['map'] = {'points': Z.round(2).tolist(), 'digits': y.tolist()}

    # 7. do a few directions keep what matters? nearest picture, by k
    order = np.random.default_rng(2).permutation(len(X))
    tr, te = order[:1300], order[1300:]
    out['guess'] = {'test_pictures': len(te), 'all_64': nearest_neighbour_right(X[tr], y[tr], X[te], y[te])}
    for k in (2, 5, 10, 20):
        Xtr, mtr = center(X[tr])
        Dk = directions(Xtr, k)
        out['guess'][k] = nearest_neighbour_right(squeeze(Xtr, Dk), y[tr], squeeze(X[te] - mtr, Dk), y[te])

    # 8. where it fails: points on a ring, squeezed to one number. One number
    # (the angle) is enough to say where a point is, but no straight line is:
    # points on opposite sides of the ring get the same number.
    a = np.sort(np.random.default_rng(3).uniform(0, 2 * np.pi, 400))
    P = np.c_[np.cos(a), np.sin(a)] * 10 + np.random.default_rng(4).normal(0, 0.3, (400, 2))
    Pc, pm = center(P)
    Dp = directions(Pc, 1)
    zp = squeeze(Pc, Dp)[:, 0]
    Pr = rebuild(zp[:, None], Dp, pm)
    along = a * 10                                   # distance along the ring
    ring = np.c_[np.cos(a), np.sin(a)] * 10           # rebuilt from the angle alone
    angle_error = float(np.sqrt(((ring - P) ** 2).sum(1)).mean())
    best = (0, 0, 0)
    for i in range(400):
        for j in range(i + 1, 400):
            d = np.linalg.norm(P[i] - P[j])
            if abs(zp[i] - zp[j]) < 0.05 and d > best[0]:
                best = (d, i, j)
    _, i, j = best
    out['curve'] = {'points': P.round(2).tolist(), 'rebuilt': Pr.round(2).tolist(), 'direction': Dp[0].tolist(),
                    'numbers': zp.round(2).tolist(),
                    'kept': spread_along(Pc, Dp[0]) / float((Pc ** 2).sum(1).mean()),
                    'mean_error': float(np.sqrt(((Pr - P) ** 2).sum(1)).mean()), 'radius': 10,
                    'angle_error': angle_error,
                    'pair': {'i': i, 'j': j, 'numbers': [float(zp[i]), float(zp[j])],
                             'apart_along_ring': float(min(along[j] - along[i], 20 * np.pi - (along[j] - along[i]))),
                             'apart_straight': float(np.linalg.norm(P[i] - P[j]))}}
    out['seconds'] = time.time() - t0
    json.dump(out, open('files/results.json', 'w'))
    p = {k: v for k, v in out.items() if k not in ('map', 'rebuild', 'directions_as_pictures', 'curve', 'one', 'two')}
    print('two', {k: v for k, v in out['two'].items() if k != 'points'})
    print(json.dumps(p, indent=1))
    print('rebuild error', {k: round(out['rebuild'][k]['error'], 1) for k in KS}, 'none', round(total, 1))
    print('curve', out['curve']['kept'], out['curve']['mean_error'], out['curve']['direction'], out['curve']['angle_error'], out['curve']['pair'])


if __name__ == '__main__':
    main()
