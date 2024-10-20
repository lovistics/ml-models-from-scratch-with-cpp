// include/models/LinearRegression.hpp
#pragma once

#include "Model.hpp"

namespace ml {
namespace models {

class LinearRegression : public Model {
public:
    LinearRegression(bool fitIntercept = true);
    ~LinearRegression() override = default;

    bool train(const utils::Matrix& features,
              const std::vector<double>& targets) override;

    std::vector<double> predict(const utils::Matrix& features) const override;
    std::vector<double> getParameters() const override;

private:
    std::vector<double> coefficients_;
    bool fitIntercept_;
};

} // namespace models
} // namespace ml