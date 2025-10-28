#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "utils/data_generation.h"
#include "utils/metrics.h"

using namespace std;

vector<double> normal_equation_fit(const Dataset& data) {
    //////////////////////////////////////////
    //Fits linear regression weights via the normal equation.
    //y = b + w_1*x_1 + w_2*x_2 + ...
    //Computes (X^T X)^(-1) X^T y using Gauss-Jordan elimination.
    //////////////////////////////////////////
    const int n_samples = static_cast<int>(data.X.size());
    if (n_samples == 0) {
        throw invalid_argument("Dataset is empty");
    }
    const int n_features = static_cast<int>(data.X[0].size());

    // XtX and Xty for augmented matrix with bias term.
    vector<vector<double>> XtX(n_features + 1, vector<double>(n_features + 1, 0.0));
    vector<double> Xty(n_features + 1, 0.0);

    for (int i = 0; i < n_samples; ++i) {
        vector<double> row(n_features + 1, 1.0);
        for (int j = 0; j < n_features; ++j) {
            row[j + 1] = data.X[i][j];
        }

        for (int a = 0; a < n_features + 1; ++a) {
            Xty[a] += row[a] * data.y[i];
            for (int b = 0; b < n_features + 1; ++b) {
                XtX[a][b] += row[a] * row[b];
            }
        }
    }

    //Solve XtX * w = Xty using Gauss-Jordan elimination.
    const double eps = 1e-12;
    vector<vector<double>> augmented(n_features + 1, vector<double>(n_features + 2, 0.0));
    for (int i = 0; i < n_features + 1; ++i) {
        for (int j = 0; j < n_features + 1; ++j) {
            augmented[i][j] = XtX[i][j];
        }
        augmented[i][n_features + 1] = Xty[i];
    }

    for (int col = 0; col < n_features + 1; ++col) {
        int pivot_row = col;
        double max_val = fabs(augmented[col][col]);
        for (int row = col + 1; row < n_features + 1; ++row) {
            const double current_val = fabs(augmented[row][col]);
            if (current_val > max_val) {
                max_val = current_val;
                pivot_row = row;
            }
        }

        if (max_val < eps) {
            throw runtime_error("Design matrix is singular; cannot solve normal equation");
        }

        if (pivot_row != col) {
            swap(augmented[pivot_row], augmented[col]);
        }

        const double pivot = augmented[col][col];
        for (int j = col; j < n_features + 2; ++j) {
            augmented[col][j] /= pivot;
        }

        for (int row = 0; row < n_features + 1; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = augmented[row][col];
            for (int j = col; j < n_features + 2; ++j) {
                augmented[row][j] -= factor * augmented[col][j];
            }
        }
    }

    vector<double> weights(n_features + 1);
    for (int i = 0; i < n_features + 1; ++i) {
        weights[i] = augmented[i][n_features + 1];
    }
    return weights;
}

vector<double> predict(const Dataset& data, const vector<double>& weights) {
    //////////////////////////////////////////
    //Applies learned weights (bias + feature coefficients)
    //to produce predictions for each sample.
    //////////////////////////////////////////
    const int n_samples = static_cast<int>(data.X.size());
    const int n_features = static_cast<int>(weights.size()) - 1;
    vector<double> predictions(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double value = weights[0];  // bias term
        for (int j = 0; j < n_features; ++j) {
            value += weights[j + 1] * data.X[i][j];
        }
        predictions[i] = value;
    }

    return predictions;
}

int main() {
    //////////////////////////////////////////
    //Entry point: generate synthetic data, fit model,
    //print learned parameters and quality metric.
    //////////////////////////////////////////
    Dataset data = get_data(200, 3, 5.0, 10.0, 2.5);

    vector<double> weights = normal_equation_fit(data);
    vector<double> y_pred = predict(data, weights);

    cout << "Learned bias: " << weights[0] << endl;
    cout << "Learned weights:";
    for (size_t i = 1; i < weights.size(); ++i) {
        cout << " " << weights[i];
    }
    cout << endl;

    cout << "MSE = " << MSE(data.y, y_pred) << endl;

    return 0;
}
