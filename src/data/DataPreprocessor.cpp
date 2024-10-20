#include "../../include/data/DataPreprocessor.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cmath>

namespace ml {
namespace data {

std::pair<std::pair<utils::Matrix, std::vector<double>>,
          std::pair<utils::Matrix, std::vector<double>>>
DataPreprocessor::trainTestSplit(const utils::Matrix& features,
                                 const std::vector<double>& targets,
                                 double trainRatio,
                                 bool shuffle) {
    if (features.rows() != targets.size()) {
        throw std::invalid_argument("Number of samples in features and targets must match");
    }

    size_t numSamples = features.rows();
    size_t numTrainSamples = static_cast<size_t>(numSamples * trainRatio);

    std::vector<size_t> indices(numSamples);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }

    utils::Matrix trainFeatures(numTrainSamples, features.cols());
    std::vector<double> trainTargets(numTrainSamples);
    utils::Matrix testFeatures(numSamples - numTrainSamples, features.cols());
    std::vector<double> testTargets(numSamples - numTrainSamples);

    for (size_t i = 0; i < numTrainSamples; ++i) {
        trainFeatures[i] = features[indices[i]];
        trainTargets[i] = targets[indices[i]];
    }

    for (size_t i = numTrainSamples; i < numSamples; ++i) {
        testFeatures[i - numTrainSamples] = features[indices[i]];
        testTargets[i - numTrainSamples] = targets[indices[i]];
    }

    return {{trainFeatures, trainTargets}, {testFeatures, testTargets}};
}

utils::Matrix DataPreprocessor::standardize(const utils::Matrix& features) {
    utils::Matrix standardizedFeatures(features.rows(), features.cols());

    for (size_t j = 0; j < features.cols(); ++j) {
        double mean = 0.0;
        double variance = 0.0;

        // Calculate mean
        for (size_t i = 0; i < features.rows(); ++i) {
            mean += features[i][j];
        }
        mean /= features.rows();

        // Calculate variance
        for (size_t i = 0; i < features.rows(); ++i) {
            double diff = features[i][j] - mean;
            variance += diff * diff;
        }
        variance /= features.rows();

        double stdDev = std::sqrt(variance);

        // Standardize
        for (size_t i = 0; i < features.rows(); ++i) {
            standardizedFeatures[i][j] = (features[i][j] - mean) / stdDev;
        }
    }

    return standardizedFeatures;
}

utils::Matrix DataPreprocessor::normalize(const utils::Matrix& features) {
    utils::Matrix normalizedFeatures(features.rows(), features.cols());

    for (size_t j = 0; j < features.cols(); ++j) {
        double minVal = features[0][j];
        double maxVal = features[0][j];

        // Find min and max
        for (size_t i = 1; i < features.rows(); ++i) {
            minVal = std::min(minVal, features[i][j]);
            maxVal = std::max(maxVal, features[i][j]);
        }

        double range = maxVal - minVal;

        // Normalize
        for (size_t i = 0; i < features.rows(); ++i) {
            normalizedFeatures[i][j] = (features[i][j] - minVal) / range;
        }
    }

    return normalizedFeatures;
}

utils::Matrix DataPreprocessor::addBias(const utils::Matrix& features) {
    utils::Matrix biasedFeatures(features.rows(), features.cols() + 1);

    for (size_t i = 0; i < features.rows(); ++i) {
        biasedFeatures[i][0] = 1.0;  // Bias term
        for (size_t j = 0; j < features.cols(); ++j) {
            biasedFeatures[i][j + 1] = features[i][j];
        }
    }

    return biasedFeatures;
}

} // namespace data
} // namespace ml