#include "math.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

vector<vector<double>> add_bias_column(const vector<vector<double>>& X) {
    vector<vector<double>> with_bias;
    with_bias.reserve(X.size());
    for (const auto& row : X) {
        vector<double> extended;
        extended.reserve(row.size() + 1);
        extended.push_back(1.0);  // bias term
        extended.insert(extended.end(), row.begin(), row.end());
        with_bias.push_back(std::move(extended));
    }
    return with_bias;
}

vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    if (matrix.empty()) {
        return {};
    }
    const size_t rows = matrix.size();
    const size_t cols = matrix.front().size();
    vector<vector<double>> result(cols, vector<double>(rows, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        if (matrix[i].size() != cols) {
            throw invalid_argument("Matrix must be rectangular for transpose");
        }
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

double dot(const vector<double>& lhs, const vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw invalid_argument("Dot product requires vectors of equal length");
    }

    double result = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result += lhs[i] * rhs[i];
    }

    return result;
}

vector<vector<double>> matmul(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    if (A.empty() || B.empty()) {
        throw invalid_argument("Matrices must be non-empty for multiplication");
    }
    const size_t a_rows = A.size();
    const size_t a_cols = A.front().size();
    const size_t b_rows = B.size();
    const size_t b_cols = B.front().size();

    if (a_cols != b_rows) {
        throw invalid_argument("Incompatible shapes for matrix multiplication");
    }

    for (const auto& row : A) {
        if (row.size() != a_cols) {
            throw invalid_argument("Matrix A must be rectangular");
        }
    }
    for (const auto& row : B) {
        if (row.size() != b_cols) {
            throw invalid_argument("Matrix B must be rectangular");
        }
    }

    vector<vector<double>> result(a_rows, vector<double>(b_cols, 0.0));
    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t k = 0; k < a_cols; ++k) {
            const double a_val = A[i][k];
            if (a_val == 0.0) {
                continue;
            }
            for (size_t j = 0; j < b_cols; ++j) {
                result[i][j] += a_val * B[k][j];
            }
        }
    }
    return result;
}

vector<double> matvec(const vector<vector<double>>& A, const vector<double>& x) {
    if (A.empty()) {
        return {};
    }
    const size_t rows = A.size();
    const size_t cols = A.front().size();
    if (cols != x.size()) {
        throw invalid_argument("Matrix columns must match vector size");
    }

    vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        if (A[i].size() != cols) {
            throw invalid_argument("Matrix must be rectangular");
        }
        result[i] = dot(A[i], x);
    }
    return result;
}

vector<double> solve_gauss_jordan(vector<vector<double>> A, vector<double> b) {
    if (A.empty()) {
        throw invalid_argument("Matrix A must be non-empty");
    }
    const size_t n = A.size();
    if (A.front().size() != n) {
        throw invalid_argument("Matrix A must be square for Gauss-Jordan elimination");
    }
    if (b.size() != n) {
        throw invalid_argument("Vector b size must match matrix dimensions");
    }

    vector<vector<double>> augmented(n, vector<double>(n + 1, 0.0));
    for (size_t i = 0; i < n; ++i) {
        if (A[i].size() != n) {
            throw invalid_argument("Matrix A must be square");
        }
        for (size_t j = 0; j < n; ++j) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][n] = b[i];
    }

    const double eps = 1e-12;
    for (size_t col = 0; col < n; ++col) {
        size_t pivot_row = col;
        double max_val = fabs(augmented[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            const double current_val = fabs(augmented[row][col]);
            if (current_val > max_val) {
                max_val = current_val;
                pivot_row = row;
            }
        }

        if (max_val < eps) {
            throw runtime_error("Matrix is singular; cannot solve system");
        }

        if (pivot_row != col) {
            swap(augmented[pivot_row], augmented[col]);
        }

        const double pivot = augmented[col][col];
        for (size_t j = col; j < n + 1; ++j) {
            augmented[col][j] /= pivot;
        }

        for (size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = augmented[row][col];
            for (size_t j = col; j < n + 1; ++j) {
                augmented[row][j] -= factor * augmented[col][j];
            }
        }
    }

    vector<double> solution(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        solution[i] = augmented[i][n];
    }
    return solution;
}

vector<vector<double>> invert(vector<vector<double>> A) {
    if (A.empty()) {
        throw invalid_argument("Matrix must be non-empty to invert");
    }
    const size_t n = A.size();
    if (A.front().size() != n) {
        throw invalid_argument("Matrix must be square to invert");
    }

    vector<vector<double>> augmented(n, vector<double>(2 * n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        if (A[i].size() != n) {
            throw invalid_argument("Matrix must be square to invert");
        }
        for (size_t j = 0; j < n; ++j) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][n + i] = 1.0;
    }

    const double eps = 1e-12;
    for (size_t col = 0; col < n; ++col) {
        size_t pivot_row = col;
        double max_val = fabs(augmented[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            const double current_val = fabs(augmented[row][col]);
            if (current_val > max_val) {
                max_val = current_val;
                pivot_row = row;
            }
        }

        if (max_val < eps) {
            throw runtime_error("Matrix is singular; cannot invert");
        }

        if (pivot_row != col) {
            swap(augmented[pivot_row], augmented[col]);
        }

        const double pivot = augmented[col][col];
        for (size_t j = col; j < 2 * n; ++j) {
            augmented[col][j] /= pivot;
        }

        for (size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = augmented[row][col];
            for (size_t j = col; j < 2 * n; ++j) {
                augmented[row][j] -= factor * augmented[col][j];
            }
        }
    }

    vector<vector<double>> inverse(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inverse[i][j] = augmented[i][n + j];
        }
    }
    return inverse;
}
