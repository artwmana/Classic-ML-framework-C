#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>

std::vector<std::vector<double>> add_bias_column(const std::vector<std::vector<double>>& X);
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix);
double dot(const std::vector<double>& lhs, const std::vector<double>& rhs);
std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
std::vector<double> matvec(const std::vector<std::vector<double>>& A, const std::vector<double>& x);
std::vector<double> solve_gauss_jordan(std::vector<std::vector<double>> A, std::vector<double> b);
std::vector<std::vector<double>> invert(std::vector<std::vector<double>> A);

#endif
