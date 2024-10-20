#include "../../include/models/KNNClassifier.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <map>

namespace ml {
namespace models {

KNNClassifier::KNNClassifier(size_t k) : k_(k) {}

bool KNNClassifier::train(const utils::Matrix& features,
                          const std::vector<double>& targets) {
    if (features.rows() != targets.size()) {
        throw std::invalid_argument("Number of samples in features and targets must match");
    }

    trainFeatures_ = features;
    trainTargets_ = targets;

    return true;
}

std::vector<double> KNNClassifier::predict(const utils::Matrix& features) const {
    std::vector<double> predictions(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        std::vector<std::pair<double, double>> distances;
        for (size_t j = 0; j < trainFeatures_.rows(); ++j) {
            double distance = euclideanDistance(features[i], trainFeatures_[j]);
            distances.emplace_back(distance, trainTargets_[j]);
        }

        std::partial_sort(distances.begin(), distances.begin() + k_, distances.end());

        std::map<double, int> classVotes;
        for (size_t j = 0; j < k_; ++j) {
            ++classVotes[distances[j].second];
        }

        auto maxVote = std::max_element(
            classVotes.begin(), classVotes.end(),
            [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.second < p2.second;
            });

        predictions[i] = maxVote->first;
    }

    return predictions;
}

std::vector<double> KNNClassifier::getParameters() const {
    // KNN doesn't have parameters in the traditional sense
    return {};
}

double KNNClassifier::euclideanDistance(const std::vector<double>& a,
                                        const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

} // namespace models
} // namespace ml