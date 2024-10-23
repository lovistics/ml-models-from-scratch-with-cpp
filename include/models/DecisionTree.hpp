#pragma once

#include "Model.hpp"
#include <memory>

namespace ml {
namespace models {

class DecisionTreeNode {
public:
    DecisionTreeNode(double value);
    ~DecisionTreeNode() = default;

    std::unique_ptr<DecisionTreeNode> left;
    std::unique_ptr<DecisionTreeNode> right;
    size_t featureIndex;
    double threshold;
    double value;
};

class DecisionTree : public Model {
public:
    explicit DecisionTree(size_t maxDepth = 5,
                         size_t minSamplesSplit = 2,
                         size_t maxFeatures = 0);
    ~DecisionTree() override = default;

    bool train(const utils::Matrix& features,
              const std::vector<double>& targets) override;

    std::vector<double> predict(const utils::Matrix& features) const override;
    std::vector<double> getParameters() const override;

private:
    std::unique_ptr<DecisionTreeNode> root_;
    size_t maxDepth_;
    size_t minSamplesSplit_;
    size_t maxFeatures_;

    std::unique_ptr<DecisionTreeNode> buildTree(const utils::Matrix& features,
                                              const std::vector<double>& targets,
                                              size_t depth);

    static double calculateGini(const std::vector<double>& targets);
    static std::pair<double, double> findBestSplit(const utils::Matrix& features,
                                                  const std::vector<double>& targets,
                                                  size_t featureIndex);
};

} // namespace models
} // namespace ml