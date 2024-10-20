#include "../../include/utils/Metrics.hpp"
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>

namespace ml {
namespace utils {

double Metrics::meanSquaredError(const std::vector<double>& actual,
                               const std::vector<double>& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }

    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = actual[i] - predicted[i];
        sum += diff * diff;
    }

    return sum / actual.size();
}

double Metrics::rootMeanSquaredError(const std::vector<double>& actual,
                                   const std::vector<double>& predicted) {
    return std::sqrt(meanSquaredError(actual, predicted));
}

double Metrics::accuracy(const std::vector<double>& actual,
                       const std::vector<double>& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }

    size_t correct = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - predicted[i]) < 1e-10) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / actual.size();
}

Matrix Metrics::confusionMatrix(const std::vector<double>& actual,
                              const std::vector<double>& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }

    // Find unique classes
    std::vector<double> classes;
    for (double val : actual) {
        if (std::find(classes.begin(), classes.end(), val) == classes.end()) {
            classes.push_back(val);
        }
    }
    for (double val : predicted) {
        if (std::find(classes.begin(), classes.end(), val) == classes.end()) {
            classes.push_back(val);
        }
    }
    std::sort(classes.begin(), classes.end());

    // Create mapping from class value to index
    std::unordered_map<double, size_t> classToIndex;
    for (size_t i = 0; i < classes.size(); ++i) {
        classToIndex[classes[i]] = i;
    }

    // Initialize confusion matrix
    Matrix confMatrix(classes.size(), classes.size());

    // Fill confusion matrix
    for (size_t i = 0; i < actual.size(); ++i) {
        size_t actualIdx = classToIndex[actual[i]];
        size_t predictedIdx = classToIndex[predicted[i]];
        confMatrix[actualIdx][predictedIdx] += 1.0;
    }

    return confMatrix;
}

double Metrics::rSquared(const std::vector<double>& actual,
                        const std::vector<double>& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }

    // Calculate mean of actual values
    double mean = 0.0;
    for (double val : actual) {
        mean += val;
    }
    mean /= actual.size();

    // Calculate total sum of squares and residual sum of squares
    double totalSS = 0.0;
    double residualSS = 0.0;
    
    for (size_t i = 0; i < actual.size(); ++i) {
        double diffFromMean = actual[i] - mean;
        totalSS += diffFromMean * diffFromMean;
        
        double residual = actual[i] - predicted[i];
        residualSS += residual * residual;
    }

    // Handle edge case where all actual values are the same
    if (totalSS < 1e-10) {
        return 1.0;
    }

    return 1.0 - (residualSS / totalSS);
}

} // namespace utils
} // namespace ml