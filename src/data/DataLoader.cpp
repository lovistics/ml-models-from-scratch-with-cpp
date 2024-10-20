#include "../../include/data/DataLoader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace ml {
namespace data {

bool DataLoader::loadFromCSV(const std::string& filepath, bool hasHeader, char delimiter) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    std::vector<std::vector<double>> featuresData;
    targets_.clear();
    featureNames_.clear();

    // Handle header if present
    if (hasHeader && std::getline(file, line)) {
        featureNames_ = parseLine(line, delimiter);
        // Remove the target column name
        featureNames_.pop_back();
    }

    // Read data
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = parseLine(line, delimiter);
        if (tokens.empty()) continue;

        // Last column is the target
        try {
            targets_.push_back(std::stod(tokens.back()));
            tokens.pop_back();

            std::vector<double> featureRow;
            featureRow.reserve(tokens.size());
            
            for (const auto& token : tokens) {
                featureRow.push_back(std::stod(token));
            }
            
            featuresData.push_back(std::move(featureRow));
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing line: " + line + "\nError: " + e.what());
        }
    }

    if (featuresData.empty()) {
        return false;
    }

    // Convert to Matrix
    features_ = utils::Matrix(featuresData);
    
    return true;
}

std::vector<std::string> DataLoader::parseLine(const std::string& line, char delimiter) const {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    return tokens;
}

} // namespace data
} // namespace ml


