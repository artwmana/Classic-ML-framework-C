#include <iostream>
#include <random>
using namespace std;

struct Dataset {
    vector<double> X;
    vector<double> y;
};

Dataset get_data(const int n_samples, const float noise, const double bias, const float real_w, const int n_features=1, const int random_state=42){
    //y = X * w + noise
    Dataset data;
    data.X.resize(n_samples);
    data.y.resize(n_samples);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist_x(-10, 10);
    normal_distribution<> dist_noise(0, noise);

    for(int i = 0; i < n_samples; i++){
        data.X[i] = dist_x(gen);
        double noise = dist_noise(gen);
        data.y[i] = real_w * data.X[i] + bias + noise;
    }

    return data;
}

int main() {
    Dataset data = get_data(200, 15, 5, 3.5);

    for (int i = 0; i < 10; i++) {
        cout << "x = " << data.X[i]
             << ", y = " << data.y[i] << endl;
    }

    return 0;
}