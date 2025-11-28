#!/usr/bin/env bash
set -euo pipefail

g++ -std=c++17 \
  linear_regression.cpp \
  utils/data_generation.cpp \
  utils/metrix.cpp \
  utils/math.cpp \
  -o linear_regression
