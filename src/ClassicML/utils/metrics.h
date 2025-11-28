#ifndef METRICS_H
#define METRICS_H

#include <vector>

double MSE(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double MAE(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double R2(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double MAPE(const std::vector<double>& y_true, const std::vector<double>& y_pred);

#endif 