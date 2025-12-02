# Classic ML framework (C++)
Minimal classic machine learning playground in C++17 featuring a from‑scratch linear regression implementation, basic metrics, and small data utilities. It is intended as an educational reference and a starting point for experimenting with classic ML algorithms without external dependencies.

## Features
- Linear regression trained via closed-form normal equation and batch gradient descent
- Metrics: MSE, MAE, MAPE, R²
- Lightweight matrix helpers (transpose, matmul, inversion) with error checking
- Synthetic dataset generator plus CSV loader for custom data

## Project layout
- `src/ClassicML/linear_regression.cpp` — demo executable: trains, evaluates, and prints metrics
- `src/ClassicML/utils/` — math helpers, metrics, and data utilities (`data_generation.*`, `math.*`, `metrics.h`, `metrix.cpp`)
- `build.sh` — convenience compile script (expects to be run from `src/ClassicML`)

## Build and run
The code targets C++17 and uses only the standard library.

```bash
# From the repo root, compile the demo binary
g++ -std=c++17 \
  src/ClassicML/linear_regression.cpp \
  src/ClassicML/utils/data_generation.cpp \
  src/ClassicML/utils/metrix.cpp \
  src/ClassicML/utils/math.cpp \
  -o linear_regression

# Run (uses synthetic data by default)
./linear_regression
```

Alternatively, you can run `build.sh` from inside `src/ClassicML`:

```bash
cd src/ClassicML
../../build.sh
./linear_regression
```

## Dataset options
- Synthetic: default run generates a linear dataset with configurable samples, feature count, noise, bias, weight, and seed (see `get_data` in `data_generation.cpp`).
- CSV: use `load_dataset_from_csv(path, target_first)` to feed your own data if you extend the demo to accept file paths; each row should contain features plus a target column.

## Extending
- Swap in other optimizers or loss functions using the math helpers.
- Add more metrics by following the style of `metrix.cpp`.
- Wire the CSV loader into `main` if you want to benchmark on real datasets.

## License
MIT — see `LICENSE`.
