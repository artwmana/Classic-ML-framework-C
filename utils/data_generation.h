#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <string>
#include <vector>

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

Dataset get_data(int n_samples, int n_features, double noise, double bias, double real_w, int random_state = 42);
Dataset load_dataset_from_csv(const std::string& path, bool target_first);

#endif
