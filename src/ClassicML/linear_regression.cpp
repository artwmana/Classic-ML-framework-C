#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "utils/data_generation.h"
#include "utils/math.h"
#include "utils/metrics.h"

using namespace std;

vector<double> normal_equation_fit(const Dataset& data) {
    //////////////////////////////////////////
    //Fits linear regression weights via the normal equation.
    //w = (X^T * X)^-1 * X.T * y
    //after algebraic simplification.
    //////////////////////////////////////////
    if (data.X.empty()) {
        throw invalid_argument("Dataset is empty");
    }
    if (data.X[0].empty()) {
        throw invalid_argument("Dataset must contain at least one feature");
    }

    const vector<vector<double>> X_design = add_bias_column(data.X);
    const vector<vector<double>> Xt = transpose(X_design);
    const vector<vector<double>> XtX = matmul(Xt, X_design);
    const vector<vector<double>> XtX_inv = invert(XtX);
    const vector<double> XtY = matvec(Xt, data.y);

    return matvec(XtX_inv, XtY);
}

vector<double> gradient_descent_fit(const Dataset& data, double learning_rate, int max_iters, double tolerance) {
    //////////////////////////////////////////
    //Fits linear regression weights by minimizing MSE
    //with batch gradient descent.
    //////////////////////////////////////////
    const size_t n_samples = data.X.size();
    if (n_samples == 0) {
        throw invalid_argument("Dataset is empty");
    }
    const size_t n_features = data.X[0].size();
    if (n_features == 0) {
        throw invalid_argument("Dataset must contain at least one feature");
    }

    const vector<vector<double>> X_design = add_bias_column(data.X);
    const vector<vector<double>> Xt = transpose(X_design);
    vector<double> weights(n_features + 1, 0.0);

    double previous_loss = numeric_limits<double>::infinity();
    for (int iter = 0; iter < max_iters; ++iter) {
        vector<double> predictions = matvec(X_design, weights);
        vector<double> errors(n_samples, 0.0);

        double loss = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            errors[i] = predictions[i] - data.y[i];
            loss += errors[i] * errors[i];
        }
        loss /= static_cast<double>(n_samples);

        if (fabs(previous_loss - loss) < tolerance) {
            break;
        }
        previous_loss = loss;

        vector<double> gradient = matvec(Xt, errors);
        const double scale = 2.0 / static_cast<double>(n_samples);

        double max_update = 0.0;
        for (size_t j = 0; j < gradient.size(); ++j) {
            gradient[j] *= scale;
            const double update = learning_rate * gradient[j];
            weights[j] -= update;
            max_update = max(max_update, fabs(update));
        }

        if (max_update < tolerance) {
            break;
        }
    }

    return weights;
}

vector<double> predict(const Dataset& data, const vector<double>& weights) {
    //////////////////////////////////////////
    //Applies learned weights (bias + feature coefficients)
    //to produce predictions for each sample.
    //////////////////////////////////////////
    if (weights.empty()) {
        throw invalid_argument("Weights vector is empty");
    }
    if (data.X.empty()) {
        return {};
    }
    if (weights.size() != data.X[0].size() + 1) {
        throw invalid_argument("Weights size must equal number of features plus bias");
    }

    const vector<vector<double>> design = add_bias_column(data.X);
    return matvec(design, weights);
}

namespace {
bool file_exists(const string& path) {
    ifstream in(path);
    return in.good();
}

string resolve_dataset_path(const string& candidate, const char* argv0) {
    if (file_exists(candidate)) {
        return candidate;
    }
    if (argv0 != nullptr) {
        string exe_path(argv0);
        const size_t slash = exe_path.find_last_of("/\\");
        if (slash != string::npos) {
            string alternative = exe_path.substr(0, slash + 1) + candidate;
            if (file_exists(alternative)) {
                return alternative;
            }
        }
    }
    return candidate;
}
}  // namespace

int main(int argc, char* argv[]) {
    //////////////////////////////////////////
    //Entry point: generate synthetic data, fit model
    //with two approaches, and report metrics.
    //////////////////////////////////////////
    const string default_path = "Classic-ML-framework-C/data/dataset.csv";
    const string dataset_path = resolve_dataset_path((argc > 1) ? argv[1] : default_path, argv[0]);
    Dataset data;
    try {
        data = get_data(100, 2, 0.1, 2, 3.5, 42);
    } catch (const exception& e) {
        cerr << "Failed to load dataset from " << dataset_path << ": " << e.what() << endl;
        cerr << "Ensure the Python notebook exports a CSV with target in the first column and features after it." << endl;
        return 1;
    }

    const double learning_rate = 0.01;
    const int max_iters = 20000;
    const double tolerance = 1e-8;

    vector<double> normal_weights = normal_equation_fit(data);
    vector<double> normal_pred = predict(data, normal_weights);

    vector<double> gd_weights = gradient_descent_fit(data, learning_rate, max_iters, tolerance);
    vector<double> gd_pred = predict(data, gd_weights);

    auto print_weights = [](const vector<double>& weights) {
        cout << "bias = " << weights.front() << ", weights:";
        for (size_t i = 1; i < weights.size(); ++i) {
            cout << " " << weights[i];
        }
        cout << endl;
    };

    auto print_metrics = [&](const vector<double>& preds) {
        cout << "MSE = " << MSE(data.y, preds)
             << ", MAE = " << MAE(data.y, preds)
             << ", MAPE = " << MAPE(data.y, preds)
             << ", R2 = " << R2(data.y, preds) << endl;
    };

    cout << fixed << setprecision(4);

    cout << "Normal equation solution:" << endl;
    print_weights(normal_weights);
    print_metrics(normal_pred);

    cout << "\nGradient descent solution:" << endl;
    print_weights(gd_weights);
    print_metrics(gd_pred);

    return 0;
}
