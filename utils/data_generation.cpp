#include "data_generation.h"
#include <random>
using namespace std;

Dataset get_data(int n_samples, int n_features, double noise, double bias, double real_w, const int random_state){
    //////////////////////////////////////////
    // Generates synthetic linear regression data.
    // n_samples  -> number of rows in X.
    // n_features -> number of features per sample.
    // noise      -> standard deviation of Gaussian noise.
    // bias       -> intercept added to every target.
    // real_w     -> weight value shared across features.
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
