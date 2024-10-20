// include/models/Model.hpp
#pragma once

#include <vector>
#include <memory>
#include "../utils/Matrix.hpp"

namespace ml {
namespace models {

class Model {
public:
    virtual ~Model() = default;

    /**
     * @brief Train the model
     * @param features Training features
     * @param targets Training targets
     * @return True if training was successful
     */
    virtual bool train(const utils::Matrix& features,
                      const std::vector<double>& targets) = 0;

    /**
     * @brief Make predictions using the trained model
     * @param features Input features
     * @return Vector of predictions
     */
    virtual std::vector<double> predict(const utils::Matrix& features) const = 0;

    /**
     * @brief Get the model parameters
     * @return Vector of model parameters
     */
    virtual std::vector<double> getParameters() const = 0;

protected:
    Model() = default;
};

} // namespace models
} // namespace ml