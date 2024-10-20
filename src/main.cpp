#include <iostream>
#include <memory>
#include "../include/data/DataLoader.hpp"
#include "../include/data/DataPreprocessor.hpp"
#include "../include/models/LinearRegression.hpp"
#include "../include/models/LogisticRegression.hpp"
#include "../include/models/KNNClassifier.hpp"
#include "../include/models/DecisionTree.hpp"
#include "../include/utils/Metrics.hpp"

using namespace ml;

void evaluateModel(const std::string& modelName,
                  const utils::Matrix& testFeatures,
                  const std::vector<double>& testTargets,
                  const std::vector<double>& predictions) {
    std::cout << "\nEvaluating " << modelName << ":\n";
    std::cout << "MSE: " << utils::Metrics::meanSquaredError(testTargets, predictions) << "\n";
    std::cout << "RMSE: " << utils::Metrics::rootMeanSquaredError(testTargets, predictions) << "\n";
    std::cout << "R²: " << utils::Metrics::rSquared(testTargets, predictions) << "\n";
    std::cout << "Accuracy: " << utils::Metrics::accuracy(testTargets, predictions) << "\n";
}

int main() {
    try {
        // Load data
        data::DataLoader loader;
        if (!loader.loadFromCSV("data/wine.csv")) {
            std::cerr << "Failed to load data\n";
            return 1;
        }

        // Preprocess data
        auto features = loader.getFeatures();
        auto targets = loader.getTargets();
        
        // Standardize features
        features = data::DataPreprocessor::standardize(features);
        
        // Split data
        auto [trainData, testData] = data::DataPreprocessor::trainTestSplit(features, targets, 0.8);
        auto& [trainFeatures, trainTargets] = trainData;
        auto& [testFeatures, testTargets] = testData;

        // Train and evaluate Linear Regression
        {
            models::LinearRegression model;
            if (model.train(trainFeatures, trainTargets)) {
                auto predictions = model.predict(testFeatures);
                evaluateModel("Linear Regression", testFeatures, testTargets, predictions);
            }
        }

        // Train and evaluate Logistic Regression
        {
            models::LogisticRegression model;
            if (model.train(trainFeatures, trainTargets)) {
                auto predictions = model.predict(testFeatures);
                evaluateModel("Logistic Regression", testFeatures, testTargets, predictions);
            }
        }

        // Train and evaluate KNN Classifier
        {
            models::KNNClassifier model(5);
            if (model.train(trainFeatures, trainTargets)) {
                auto predictions = model.predict(testFeatures);
                evaluateModel("KNN Classifier", testFeatures, testTargets, predictions);
            }
        }

        // Train and evaluate Decision Tree
        {
            models::DecisionTree model(5);
            if (model.train(trainFeatures, trainTargets)) {
                auto predictions = model.predict(testFeatures);
                evaluateModel("Decision Tree", testFeatures, testTargets, predictions);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}