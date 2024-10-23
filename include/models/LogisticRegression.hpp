#pragma once

#include "Model.hpp"

namespace ml {
namespace models {

class LogisticRegression : public Model {
public:
    LogisticRegression(double learningRate = 0.01,
                      size_t maxIterations = 1000,
                      double tolerance = 1e-4,
                      bool fitIntercept = true);
    ~LogisticRegression() override = default;

    bool train(const utils::Matrix& features,
              const std::vector<double>& targets) override;

    std::vector<double> predict(const utils::Matrix& features) const override;
    std::vector<double> getParameters() const override;

private:
    std::vector<double> coefficients_;
    double learningRate_;
    size_t maxIterations_;
    double tolerance_;
    bool fitIntercept_;

    static double sigmoid(double x);
    double computeCost(const utils::Matrix& features,
                      const std::vector<double>& targets) const;
};

} // namespace models
} // namespace ml