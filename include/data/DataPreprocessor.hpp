#pragma once

#include <vector>
#include <memory>
#include "../utils/Matrix.hpp"

namespace ml {
namespace data {

class DataPreprocessor {
public:
    /**
     * @brief Split data into training and testing sets
     * @param features Input feature matrix
     * @param targets Input target vector
     * @param trainRatio Ratio of training data (0.0 - 1.0)
     * @param shuffle Whether to shuffle the data before splitting
     * @return Pair of training and testing data
     */
    static std::pair<std::pair<utils::Matrix, std::vector<double>>,
                    std::pair<utils::Matrix, std::vector<double>>>
    trainTestSplit(const utils::Matrix& features,
                  const std::vector<double>& targets,
                  double trainRatio = 0.8,
                  bool shuffle = true);

    /**
     * @brief Standardize features (zero mean, unit variance)
     * @param features Input feature matrix
     * @return Standardized features
     */
    static utils::Matrix standardize(const utils::Matrix& features);

    /**
     * @brief Normalize features to [0, 1] range
     * @param features Input feature matrix
     * @return Normalized features
     */
    static utils::Matrix normalize(const utils::Matrix& features);

    /**
     * @brief Add bias term (column of ones) to features
     * @param features Input feature matrix
     * @return Features with bias term
     */
    static utils::Matrix addBias(const utils::Matrix& features);

private:
    DataPreprocessor() = delete;  // Static class
};

} // namespace data
} // namespace ml