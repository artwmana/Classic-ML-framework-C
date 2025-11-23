#include <cmath>
#include <numeric>
#include <stdexcept>
#include "metrics.h"

using namespace std;

double MSE(const vector<double>& y_true, const vector<double>& y_pred){
    //////////////////////////////////////////
    //Implementation of MSE
    //MSE = 1/n ∑ (y_true - y_pred)^2
    //////////////////////////////////////////
    if (y_true.size() != y_pred.size()) {
        throw invalid_argument("Vectors must be the same length for MSE");
    }
    if (y_true.empty()) {
        throw invalid_argument("Vectors must be non-empty for MSE");
    }
    double sum = 0.0;
    for(size_t i = 0; i < y_true.size(); i++){
        double diff = y_true[i]-y_pred[i];
        sum += diff * diff;
    }

    return sum / y_true.size();
}

double MAE(const vector<double>& y_true, const vector<double>& y_pred){
    //////////////////////////////////////////
    //Implementation of Mean absolute error
    //MAE = 1/n ∑ |y_true - y_pred|
    //////////////////////////////////////////
    if (y_true.size() != y_pred.size()) {
        throw invalid_argument("Vectors must be the same length for MAE");
    }
    if (y_true.empty()) {
        throw invalid_argument("Vectors must be non-empty for MAE");
    }
    double sum = 0.0;
    for(size_t i = 0; i < y_true.size(); i++){
        sum += fabs(y_true[i] - y_pred[i]);
    }

    return sum / y_true.size();
}

double R2(const vector<double>& y_true, const vector<double>& y_pred){
    //////////////////////////////////////////
    //Implementation of coefficient of determination (R^2)
    //////////////////////////////////////////
    if (y_true.size() != y_pred.size()) {
        throw invalid_argument("Vectors must be the same length for R2");
    }
    if (y_true.empty()) {
        throw invalid_argument("Vectors must be non-empty for R2");
    }
    const double mean = accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();

    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        const double diff = y_true[i] - y_pred[i];
        ss_res += diff * diff;

        const double mean_diff = y_true[i] - mean;
        ss_tot += mean_diff * mean_diff;
    }

    if (ss_tot == 0.0) {
        return 1.0;
    }

    return 1.0 - (ss_res / ss_tot);
}

double MAPE(const vector<double>& y_true, const vector<double>& y_pred){
    //////////////////////////////////////////
    //Implementation of Mean absolute percentage error
    //MAPE = 1/n ∑ |(y_pred-y_true)/y_true|
    //////////////////////////////////////////
    if (y_true.size() != y_pred.size()) {
        throw invalid_argument("Vectors must be the same length for MAPE");
    }
    if (y_true.empty()) {
        throw invalid_argument("Vectors must be non-empty for MAPE");
    }
    double sum = 0.0;
    size_t valid_count = 0;
    for(size_t i = 0; i < y_true.size();i++){
        if (y_true[i] == 0.0) {
            continue;
        }
        double diff = fabs(y_true[i] - y_pred[i]);
        sum += diff / fabs(y_true[i]);
        ++valid_count;
    }
    if (valid_count == 0) {
        return 0.0;
    }
    return sum / static_cast<double>(valid_count);
}
