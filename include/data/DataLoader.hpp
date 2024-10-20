// include/data/DataLoader.hpp
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "../utils/Matrix.hpp"

namespace ml {
namespace data {

class DataLoader {
public:
    DataLoader() = default;
    ~DataLoader() = default;

    // Delete copy constructor and assignment operator
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    /**
     * @brief Load data from a CSV file
     * @param filepath Path to the CSV file
     * @param hasHeader Whether the CSV has a header row
     * @param delimiter CSV delimiter character
     * @return True if loading was successful
     */
    bool loadFromCSV(const std::string& filepath, 
                    bool hasHeader = true, 
                    char delimiter = ',');

    /**
     * @brief Get the loaded feature matrix
     * @return Const reference to the feature matrix
     */
    const utils::Matrix& getFeatures() const { return features_; }

    /**
     * @brief Get the loaded target vector
     * @return Const reference to the target vector
     */
    const std::vector<double>& getTargets() const { return targets_; }

    /**
     * @brief Get the feature names
     * @return Vector of feature names
     */
    const std::vector<std::string>& getFeatureNames() const { return featureNames_; }

private:
    utils::Matrix features_;
    std::vector<double> targets_;
    std::vector<std::string> featureNames_;
    
    /**
     * @brief Parse a CSV line into tokens
     * @param line Input line string
     * @param delimiter Delimiter character
     * @return Vector of tokens
     */
    std::vector<std::string> parseLine(const std::string& line, char delimiter) const;
};

} // namespace data
} // namespace ml