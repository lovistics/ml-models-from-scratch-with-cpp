#include "../../include/models/DecisionTree.hpp"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <cmath>

namespace ml {
namespace models {

DecisionTreeNode::DecisionTreeNode(double value) 
    : featureIndex(0), threshold(0.0), value(value) {}

DecisionTree::DecisionTree(size_t maxDepth, size_t minSamplesSplit, size_t maxFeatures)
    : maxDepth_(maxDepth), minSamplesSplit_(minSamplesSplit), maxFeatures_(maxFeatures) {}

bool DecisionTree::train(const utils::Matrix& features, const std::vector<double>& targets) {
    if (features.rows() != targets.size() || features.rows() == 0) {
        return false;
    }

    if (maxFeatures_ == 0) {
        maxFeatures_ = features.cols();
    }

    root_ = buildTree(features, targets, 0);
    return true;
}

std::vector<double> DecisionTree::predict(const utils::Matrix& features) const {
    std::vector<double> predictions;
    predictions.reserve(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        const DecisionTreeNode* node = root_.get();
        while (node->left && node->right) {
            if (features[i][node->featureIndex] <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        predictions.push_back(node->value);
    }

    return predictions;
}

std::vector<double> DecisionTree::getParameters() const {
    // Return empty vector as tree parameters are not easily representable as a vector
    return std::vector<double>();
}

std::unique_ptr<DecisionTreeNode> DecisionTree::buildTree(
    const utils::Matrix& features,
    const std::vector<double>& targets,
    size_t depth) {
    
    // Create leaf node if stopping criteria are met
    if (depth >= maxDepth_ || targets.size() < minSamplesSplit_) {
        double leafValue = 0.0;
        if (!targets.empty()) {
            leafValue = std::accumulate(targets.begin(), targets.end(), 0.0) / targets.size();
        }
        return std::make_unique<DecisionTreeNode>(leafValue);
    }

    // Find best split
    double bestGini = std::numeric_limits<double>::infinity();
    size_t bestFeature = 0;
    double bestThreshold = 0.0;

    std::vector<size_t> featureIndices(features.cols());
    std::iota(featureIndices.begin(), featureIndices.end(), 0);
    
    // Randomly select features if maxFeatures_ is less than total features
    if (maxFeatures_ < features.cols()) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(featureIndices.begin(), featureIndices.end(), g);
        featureIndices.resize(maxFeatures_);
    }

    for (size_t featureIdx : featureIndices) {
        auto [threshold, gini] = findBestSplit(features, targets, featureIdx);
        if (gini < bestGini) {
            bestGini = gini;
            bestFeature = featureIdx;
            bestThreshold = threshold;
        }
    }

    // If no improvement in Gini, create leaf node
    if (bestGini == std::numeric_limits<double>::infinity()) {
        double leafValue = std::accumulate(targets.begin(), targets.end(), 0.0) / targets.size();
        return std::make_unique<DecisionTreeNode>(leafValue);
    }

    // Split data
    utils::Matrix leftFeatures(0, features.cols());
    utils::Matrix rightFeatures(0, features.cols());
    std::vector<double> leftTargets, rightTargets;

    for (size_t i = 0; i < features.rows(); ++i) {
        if (features[i][bestFeature] <= bestThreshold) {
            leftFeatures = utils::Matrix(std::vector<std::vector<double>>{features[i]});
            leftTargets.push_back(targets[i]);
        } else {
            rightFeatures = utils::Matrix(std::vector<std::vector<double>>{features[i]});
            rightTargets.push_back(targets[i]);
        }
    }

    // Create node and recursively build subtrees
    auto node = std::make_unique<DecisionTreeNode>(0.0);
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(leftFeatures, leftTargets, depth + 1);
    node->right = buildTree(rightFeatures, rightTargets, depth + 1);

    return node;
}

double DecisionTree::calculateGini(const std::vector<double>& targets) {
    if (targets.empty()) return 0.0;

    std::unordered_map<double, size_t> classCounts;
    for (double target : targets) {
        ++classCounts[target];
    }

    double gini = 1.0;
    double n = static_cast<double>(targets.size());
    
    for (const auto& [_, count] : classCounts) {
        double p = count / n;
        gini -= p * p;
    }

    return gini;
}

std::pair<double, double> DecisionTree::findBestSplit(
    const utils::Matrix& features,
    const std::vector<double>& targets,
    size_t featureIndex) {
    
    std::vector<std::pair<double, double>> featureTargetPairs;
    featureTargetPairs.reserve(features.rows());
    
    for (size_t i = 0; i < features.rows(); ++i) {
        featureTargetPairs.emplace_back(features[i][featureIndex], targets[i]);
    }
    
    std::sort(featureTargetPairs.begin(), featureTargetPairs.end());
    
    double bestGini = std::numeric_limits<double>::infinity();
    double bestThreshold = 0.0;
    
    for (size_t i = 1; i < featureTargetPairs.size(); ++i) {
        if (featureTargetPairs[i].first != featureTargetPairs[i-1].first) {
            double threshold = (featureTargetPairs[i].first + featureTargetPairs[i-1].first) / 2.0;
            
            std::vector<double> leftTargets, rightTargets;
            for (const auto& pair : featureTargetPairs) {
                if (pair.first <= threshold) {
                    leftTargets.push_back(pair.second);
                } else {
                    rightTargets.push_back(pair.second);
                }
            }
            
            double gini = (leftTargets.size() * calculateGini(leftTargets) +
                         rightTargets.size() * calculateGini(rightTargets)) / features.rows();
            
            if (gini < bestGini) {
                bestGini = gini;
                bestThreshold = threshold;
            }
        }
    }
    
    return {bestThreshold, bestGini};
}

} // namespace models
} // namespace ml