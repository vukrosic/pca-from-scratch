"""Compact data for the film's pictures: reads files/results.json, adds the few
pictures run.py does not keep (the average picture, the 1,000 random spreads,
power iteration's turns on the two-pixel cloud), writes ../src/values.json."""
import json

import numpy as np
from sklearn.datasets import load_digits

from pca import center, spread_along

R = json.load(open('files/results.json'))
X, y = load_digits(return_X_y=True)
Xc, mean = center(X)
total = R['total_spread']

# the 1,000 random directions of run.py (same seed), as shares of all the spread
rng = np.random.default_rng(1)
rnd = [spread_along(Xc, v / np.linalg.norm(v)) / total for v in rng.normal(size=(1000, 64))]

# power iteration on the two-pixel cloud, from a direction far from the best one
T = X[:, [13, 21]]
Tc, _ = center(T)
C = Tc.T @ Tc / len(Tc)
d = np.array([np.cos(np.radians(150)), np.sin(np.radians(150))])
turns = [float(np.degrees(np.arctan2(d[1], d[0])))]
for _ in range(6):
    d = C @ d
    d = d / np.linalg.norm(d)
    turns.append(float(np.degrees(np.arctan2(d[1], d[0]))))

# the zero's numbers along the first six directions (for the rebuild sum)
D6 = np.array(R['directions_as_pictures'])
zero_nums = ((X[0] - mean) @ D6.T).round(2).tolist()

# the guesser on one hidden picture: its four closest pictures among the 1,300 not hidden (run.py's split)
order = np.random.default_rng(2).permutation(len(X))
tr, te = order[:1300], order[1300:]
h = te[0]
dist = ((X[tr] - X[h]) ** 2).sum(1)
near = np.argsort(dist)[:4]
guess_demo = {'hidden': X[h].astype(int).tolist(), 'digit': int(y[h]),
              'near': [{'pic': X[tr[i]].astype(int).tolist(), 'digit': int(y[tr[i]]), 'dist': float(dist[i])} for i in near]}

r = R['rebuild']
out = {
    'n': R['pictures'], 'total': total, 'one': R['one']['numbers'], 'mean': mean.round(2).tolist(),
    'cards': r['original'],
    'two': {k: R['two'][k] for k in ('pixels', 'points', 'mean', 'example', 'sweep_deg', 'sweep', 'best_deg', 'best', 'total')},
    'turns': turns,
    'random': [round(v, 4) for v in rnd],
    'line': R['line'], 'check': R['check'],
    'kept': [R['kept'][str(k)] if str(k) in R['kept'] else R['kept'][k] for k in range(1, R['directions_found'] + 1)],
    'need': R['need'], 'blank': [int(i) for i in np.where(X.std(0) == 0)[0]],
    'rebuild': {str(k): {'zero': r[str(k)]['pictures'][0], 'error': r[str(k)]['error']} for k in (1, 2, 5, 10, 20)},
    'error_none': r['error_none'],
    'dirs': R['directions_as_pictures'],
    'map': {'points': R['map']['points'], 'digits': R['map']['digits']},
    'guess': {k: R['guess'][k] for k in ('test_pictures', 'all_64', '2', '5', '10', '20')},
    'curve': R['curve'],
    'zero_nums': zero_nums, 'guess_demo': guess_demo,
}
json.dump(out, open('../src/values.json', 'w'), separators=(',', ':'))
print('turns', [round(t, 1) for t in turns], 'random max', max(rnd), 'blank', out['blank'])
