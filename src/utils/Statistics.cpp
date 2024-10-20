// src/utils/Statistics.cpp
#include "utils/Statistics.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ml {
namespace utils {

double Statistics::mean(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot calculate mean of empty vector");
    }
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double Statistics::variance(const std::vector<double>& data, int ddof) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot calculate variance of empty vector");
    }
    
    if (data.size() <= ddof) {
        throw std::invalid_argument("Not enough data points for given degrees of freedom");
    }
    
    double m = mean(data);
    double sum = 0.0;
    
    for (const auto& value : data) {
        sum += (value - m) * (value - m);
    }
    
    return sum / (data.size() - ddof);
}

double Statistics::standardDeviation(const std::vector<double>& data, int ddof) {
    return std::sqrt(variance(data, ddof));
}

Matrix Statistics::correlationMatrix(const Matrix& matrix) {
    Matrix result(matrix.cols(), matrix.cols());
    
    for (size_t i = 0; i < matrix.cols(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            std::vector<double> col_i;
            std::vector<double> col_j;
            
            for (size_t k = 0; k < matrix.rows(); ++k) {
                col_i.push_back(matrix[k][i]);
                col_j.push_back(matrix[k][j]);
            }
            
            double mean_i = mean(col_i);
            double mean_j = mean(col_j);
            double std_i = standardDeviation(col_i);
            double std_j = standardDeviation(col_j);
            
            double correlation = 0.0;
            for (size_t k = 0; k < matrix.rows(); ++k) {
                correlation += (matrix[k][i] - mean_i) * (matrix[k][j] - mean_j);
            }
            
            correlation /= (matrix.rows() - 1) * std_i * std_j;
            result[i][j] = correlation;
        }
    }
    
    return result;
}

Matrix Statistics::covarianceMatrix(const Matrix& matrix) {
    Matrix result(matrix.cols(), matrix.cols());
    
    std::vector<double> means;
    for (size_t j = 0; j < matrix.cols(); ++j) {
        std::vector<double> col;
        for (size_t i = 0; i < matrix.rows(); ++i) {
            col.push_back(matrix[i][j]);
        }
        means.push_back(mean(col));
    }
    
    for (size_t i = 0; i < matrix.cols(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < matrix.rows(); ++k) {
                sum += (matrix[k][i] - means[i]) * (matrix[k][j] - means[j]);
            }
            result[i][j] = sum / (matrix.rows() - 1);
        }
    }
    
    return result;
}

} // namespace utils
} // namespace ml