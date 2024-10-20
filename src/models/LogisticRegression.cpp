#include "../../include/models/LogisticRegression.hpp"
#include "../../include/utils/Matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace ml {
namespace models {

LogisticRegression::LogisticRegression(double learningRate, size_t maxIterations,
                                       double tolerance, bool fitIntercept)
    : learningRate_(learningRate), maxIterations_(maxIterations),
      tolerance_(tolerance), fitIntercept_(fitIntercept) {}

bool LogisticRegression::train(const utils::Matrix& features,
                               const std::vector<double>& targets) {
    if (features.rows() != targets.size()) {
        throw std::invalid_argument("Number of samples in features and targets must match");
    }

    utils::Matrix X = features;
    if (fitIntercept_) {
        X = utils::Matrix::ones(features.rows(), features.cols() + 1);
        for (size_t i = 0; i < features.rows(); ++i) {
            for (size_t j = 0; j < features.cols(); ++j) {
                X[i][j + 1] = features[i][j];
            }
        }
    }

    coefficients_ = std::vector<double>(X.cols(), 0.0);

    for (size_t iteration = 0; iteration < maxIterations_; ++iteration) {
        std::vector<double> predictions = predict(features);
        
        std::vector<double> gradient(X.cols(), 0.0);
        for (size_t i = 0; i < X.rows(); ++i) {
            double error = predictions[i] - targets[i];
            for (size_t j = 0; j < X.cols(); ++j) {
                gradient[j] += error * X[i][j];
            }
        }

        for (size_t j = 0; j < X.cols(); ++j) {
            coefficients_[j] -= learningRate_ * gradient[j] / X.rows();
        }

        double cost = computeCost(X, targets);
        if (cost < tolerance_) {
            break;
        }
    }

    return true;
}

std::vector<double> LogisticRegression::predict(const utils::Matrix& features) const {
    utils::Matrix X = features;
    if (fitIntercept_) {
        X = utils::Matrix::ones(features.rows(), features.cols() + 1);
        for (size_t i = 0; i < features.rows(); ++i) {
            for (size_t j = 0; j < features.cols(); ++j) {
                X[i][j + 1] = features[i][j];
            }
        }
    }

    std::vector<double> predictions(X.rows());
    for (size_t i = 0; i < X.rows(); ++i) {
        double z = 0.0;
        for (size_t j = 0; j < X.cols(); ++j) {
            z += X[i][j] * coefficients_[j];
        }
        predictions[i] = sigmoid(z);
    }

    return predictions;
}

std::vector<double> LogisticRegression::getParameters() const {
    return coefficients_;
}

double LogisticRegression::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double LogisticRegression::computeCost(const utils::Matrix& features,
                                       const std::vector<double>& targets) const {
    std::vector<double> predictions = predict(features);
    double cost = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        cost -= targets[i] * std::log(predictions[i]) +
                (1 - targets[i]) * std::log(1 - predictions[i]);
    }
    return cost / predictions.size();
}

} // namespace models
} // namespace ml