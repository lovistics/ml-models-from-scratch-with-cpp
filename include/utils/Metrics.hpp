#pragma once

#include <vector>
#include "../utils/Matrix.hpp"

namespace ml {
namespace utils {

class Metrics {
public:
    /**
     * @brief Calculate Mean Squared Error
     * @param actual Actual values
     * @param predicted Predicted values
     * @return MSE value
     */
    static double meanSquaredError(const std::vector<double>& actual,
                                 const std::vector<double>& predicted);

    /**
     * @brief Calculate Root Mean Squared Error
     * @param actual Actual values
     * @param predicted Predicted values
     * @return RMSE value
     */
    static double rootMeanSquaredError(const std::vector<double>& actual,
                                     const std::vector<double>& predicted);

    /**
     * @brief Calculate accuracy score
     * @param actual Actual values
     * @param predicted Predicted values
     * @return Accuracy value
     */
    static double accuracy(const std::vector<double>& actual,
                         const std::vector<double>& predicted);

    /**
     * @brief Calculate confusion matrix
     * @param actual Actual values
     * @param predicted Predicted values
     * @return Confusion matrix
     */
    static Matrix confusionMatrix(const std::vector<double>& actual,
                                const std::vector<double>& predicted);

    /**
     * @brief Calculate R-squared score
     * @param actual Actual values
     * @param predicted Predicted values
     * @return R-squared value
     */
    static double rSquared(const std::vector<double>& actual,
                         const std::vector<double>& predicted);

private:
    Metrics() = delete;  // Static class
};

} // namespace utils
} // namespace ml