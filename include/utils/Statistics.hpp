#pragma once

#include <vector>
#include "Matrix.hpp"

namespace ml {
namespace utils {

class Statistics {
public:
    /**
     * @brief Calculate mean of a vector
     * @param data Input vector
     * @return Mean value
     */
    static double mean(const std::vector<double>& data);

    /**
     * @brief Calculate variance of a vector
     * @param data Input vector
     * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
     * @return Variance value
     */
    static double variance(const std::vector<double>& data, int ddof = 1);

    /**
     * @brief Calculate standard deviation of a vector
     * @param data Input vector
     * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
     * @return Standard deviation value
     */
    static double standardDeviation(const std::vector<double>& data, int ddof = 1);

    /**
     * @brief Calculate correlation matrix
     * @param matrix Input matrix
     * @return Correlation matrix
     */
    static Matrix correlationMatrix(const Matrix& matrix);

    /**
     * @brief Calculate covariance matrix
     * @param matrix Input matrix
     * @return Covariance matrix
     */
    static Matrix covarianceMatrix(const Matrix& matrix);

private:
    Statistics() = delete;  // Static class
};

} // namespace utils
} // namespace ml