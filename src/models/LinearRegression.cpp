#include "../../include/models/LinearRegression.hpp"
#include "../../include/utils/Matrix.hpp"
#include <stdexcept>

namespace ml {
namespace models {

LinearRegression::LinearRegression(bool fitIntercept)
    : fitIntercept_(fitIntercept) {}

bool LinearRegression::train(const utils::Matrix& features,
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

    utils::Matrix y(targets.size(), 1);
    for (size_t i = 0; i < targets.size(); ++i) {
        y[i][0] = targets[i];
    }

    utils::Matrix X_T = X.transpose();
    utils::Matrix X_T_X = X_T * X;
    utils::Matrix X_T_X_inv = X_T_X.inverse();
    utils::Matrix X_T_y = X_T * y;

    utils::Matrix theta = X_T_X_inv * X_T_y;

    coefficients_.resize(theta.rows());
    for (size_t i = 0; i < theta.rows(); ++i) {
        coefficients_[i] = theta[i][0];
    }

    return true;
}

std::vector<double> LinearRegression::predict(const utils::Matrix& features) const {
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
        double prediction = 0.0;
        for (size_t j = 0; j < X.cols(); ++j) {
            prediction += X[i][j] * coefficients_[j];
        }
        predictions[i] = prediction;
    }

    return predictions;
}

std::vector<double> LinearRegression::getParameters() const {
    return coefficients_;
}

} // namespace models
} // namespace ml