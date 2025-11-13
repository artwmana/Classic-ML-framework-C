#include "data_generation.h"
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace std;

Dataset get_data(int n_samples, int n_features, double noise, double bias, double real_w, const int random_state){
    //////////////////////////////////////////
    // Generates synthetic linear regression data.
    // n_samples - number of rows in X.
    // n_features - number of features per sample.
    // noise - standard deviation of Gaussian noise.
    // bias - intercept added to every target.
    // real_w - weight value shared across features.
    // y = bias + sum_j(real_w * X_ij) + noise_i
    //////////////////////////////////////////
    Dataset data;
    data.X.resize(n_samples, vector<double>(n_features));
    data.y.resize(n_samples);

    mt19937 gen(random_state);
    uniform_real_distribution<> dist_x(-10, 10);
    normal_distribution<> dist_noise(0, noise);

    for(int i = 0; i < n_samples; i++){
        double y_val = bias;
        for(int j = 0; j < n_features; j++) {
            data.X[i][j] = dist_x(gen);
            y_val += real_w * data.X[i][j];
        }
        y_val += dist_noise(gen);
        data.y[i] = y_val;
    }

    return data;
}

namespace {
double parse_double(const string& token) {
    size_t idx = 0;
    double value = stod(token, &idx);
    if (idx != token.size()) {
        throw invalid_argument("Invalid numeric token: " + token);
    }
    return value;
}

string trim(const string& input) {
    const auto start = input.find_first_not_of(" \t\r\n");
    if (start == string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}
}  // namespace

Dataset load_dataset_from_csv(const string& path, bool target_first) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Unable to open dataset file: " + path);
    }

    Dataset data;
    string line;
    size_t expected_columns = 0;

    while (getline(file, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        vector<double> values;
        string token;
        stringstream ss(line);
        while (getline(ss, token, ',')) {
            token = trim(token);
            if (!token.empty()) {
                values.push_back(parse_double(token));
            }
        }

        if (values.size() < 2) {
            throw runtime_error("Each CSV row must contain at least one feature and one target");
        }

        if (expected_columns == 0) {
            expected_columns = values.size();
        } else if (values.size() != expected_columns) {
            throw runtime_error("Inconsistent number of columns in dataset file");
        }

        if (target_first) {
            data.y.push_back(values.front());
            data.X.emplace_back(values.begin() + 1, values.end());
        } else {
            data.X.emplace_back(values.begin(), values.end() - 1);
            data.y.push_back(values.back());
        }
    }

    if (data.X.empty()) {
        throw runtime_error("Dataset file is empty: " + path);
    }

    return data;
}
