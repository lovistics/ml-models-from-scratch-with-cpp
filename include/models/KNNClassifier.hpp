#pragma once

#include "Model.hpp"

namespace ml {
namespace models {

class KNNClassifier : public Model {
public:
    explicit KNNClassifier(size_t k = 5);
    ~KNNClassifier() override = default;

    bool train(const utils::Matrix& features,
              const std::vector<double>& targets) override;

    std::vector<double> predict(const utils::Matrix& features) const override;
    std::vector<double> getParameters() const override;

private:
    size_t k_;
    utils::Matrix trainFeatures_;
    std::vector<double> trainTargets_;

    static double euclideanDistance(const std::vector<double>& a,
                                  const std::vector<double>& b);
};

} // namespace models
} // namespace ml