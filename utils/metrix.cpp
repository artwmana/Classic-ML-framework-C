#include <numeric>
#include "metrics.h"

using namespace std;

double MSE(const vector<double>& y_true, const vector<double>& y_pred){
    //////////////////////////////////////////
    //Implementation of MSE
    //MSE = 1/n ∑ (y_true - y_pred)^2
    //////////////////////////////////////////
    double sum = 0;
    for(size_t i = 0; i < y_true.size(); i++){
        double diff = y_true[i]-y_pred[i];
        sum += diff * diff;
    }

    return sum / y_true.size();
}

double R2(const vector<double>& y_true, const vector<double>& y_pred){
    //////////////////////////////////////////
    //Implementation of coefficient of determination (R^2)
    //////////////////////////////////////////
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
