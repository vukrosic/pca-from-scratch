# PCA from scratch

Code for the video **How Can 20 Numbers Replace a Whole Picture? (PCA From Scratch)**.
Principal component analysis, PCA, finds the straight directions along which a pile of pictures differs the most. Each picture then keeps only its number along each direction. This code finds those directions with numpy alone on 1,797 hand-written digits of 8 × 8 pixels, rebuilds the pictures from a few numbers, draws the map of all pictures from two numbers, tests whether a simple guesser still tells the digits apart, and shows a ring of dots where straight directions fail.
About 350 lines of Python 3 with the checks. The method: Karl Pearson, "On lines and planes of closest fit to systems of points in space", 1901; Harold Hotelling, "Analysis of a complex of statistical variables into principal components", 1933. Eigenfaces: Matthew Turk and Alex Pentland, "Eigenfaces for Recognition", 1991.

```bash
pip install numpy scikit-learn
sh run_all.sh         # every number of the video, the exercise answer and the tests; expected output is in demo-output.txt
```

`run_all.sh` takes about fifteen seconds on a laptop CPU. The digits ship with scikit-learn (`load_digits`), so nothing else is downloaded. Nothing is trained: every result is a few matrix products.

```python
d = rng.normal(size=64)            # a random direction
for _ in range(200):
    numbers = Xc @ d               # each picture's number along d
    d = Xc.T @ numbers             # every picture pulls as hard as its number; add the pulls
    d = d / np.linalg.norm(d)      # length 1 again
```

## Words

| In the video | Usual name | In the code |
| --- | --- | --- |
| picture, 64 numbers | a data point with 64 features | a row of `X` |
| the average picture | the mean | `center` returns it |
| difference from the average | centred data | `Xc` |
| direction, its 64 weights | a unit vector, a principal axis | `d`, a row of `D` |
| a picture's number along a direction | projection, principal component score | `squeeze` |
| spread | variance along a direction | `spread_along` |
| all the spread | total variance | `total_spread` |
| pulls, again and again | power iteration | `best_direction` |
| the next direction, at a right angle | the next principal component (deflation) | `directions`, argument `against` |
| share of the spread kept | explained variance ratio | `kept` in `files/results.json` |
| rebuild | reconstruction | `rebuild` |
| rebuild error | mean squared reconstruction error | `rebuild[k].error` |
| the closest-picture guesser | 1-nearest-neighbour classifier | `run.nearest_neighbour_right` |

"Every picture pulls" and the spread table are the same step: `Xc.T @ (Xc @ d)` is 1,797 times `C @ d`, with `C = Xc.T @ Xc / 1797` the 64 × 64 table that `pca.best_direction` uses. The length step removes the factor.

## Files

| File | Idea | What the run shows |
| --- | --- | --- |
| `pca.py` | PCA with numpy only | Centre the pictures, measure the spread along a direction, find the best direction by power iteration, the next ones at right angles, squeeze and rebuild. |
| `run.py` | Every number in the video | Writes `files/results.json`. A random direction keeps 1.5 % of the spread on average, the best of 1,000 random ones 4.5 %. Power iteration: 157.9, 174.0, 176.2 … 178.9 after 200 rounds, 14.9 % of the spread, the same as numpy. 1, 2, 5, 10, 20 directions keep 14.9, 28.5, 54.5, 73.8, 89.4 %; 21 directions pass 90 %, 41 pass 99 %. Rebuild error 1,201.5 with no direction; with 1, 2, 5, 10, 20 directions 1,022.6, 858.9, 546.7, 314.5, 127.0. On 497 hidden pictures the guesser is right 99.2 % with all 64 numbers, 57.7 % with 2, 89.3 % with 5, 96.4 % with 10, 99.0 % with 20. A ring of 400 dots of radius 10: the best line keeps 52 %, rebuilt dots miss by 6.1 against 0.38 by their place around the ring; two dots 20.6 apart get the numbers −1.13 and −1.10. |
| `values.py` | Numbers for the pictures of the video | Writes `../src/values.json` for the film project. Not part of `run_all.sh`. |
| `tests.py` | Checks against numpy | 61 directions (3 pixels are blank in every picture), length 1 and at right angles, spreads equal to numpy's eigenvalues within 0.01 %, all 61 directions rebuild every picture exactly. |
| `exercise.py` | Write power iteration | See below. |
| `solution.py` | The answer | Prints `your spread: 178.9 (keeps 14.9%)` and `right!`. |

## What this is not

- **Not for bends.** PCA finds straight directions. Dots along a curve, like the ring, need more numbers than the shape has, and dots far apart can land on the same number.
- **Not about what matters.** PCA keeps the directions of most spread. A difference you care about can sit along a direction with little spread and get dropped.
- **Not the fast way.** Power iteration on a 64 × 64 table is fine here. For big photos, libraries use faster methods (the singular value decomposition, randomized solvers); they find the same directions.

## Exercise

`exercise.py`: fill in `best_direction` with the three steps above, repeated 200 times. Run `python3 exercise.py`; your spread should be 178.9, 14.9 % of all the spread. The answer is in `solution.py`.
