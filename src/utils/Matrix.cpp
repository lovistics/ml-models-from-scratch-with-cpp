// src/utils/Matrix.cpp
#include "utils/Matrix.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace ml {
namespace utils {

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, 0.0)) {}

Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty()) {
        rows_ = 0;
        cols_ = 0;
        return;
    }
    
    rows_ = data.size();
    cols_ = data[0].size();
    data_ = data;

    // Validate all rows have same length
    for (const auto& row : data) {
        if (row.size() != cols_) {
            throw std::invalid_argument("Inconsistent row sizes in input data");
        }
    }
}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = std::move(other.data_);
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
    validateDimensions(other);
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[i][j] = data_[i][j] + other.data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    validateDimensions(other);
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[i][j] = data_[i][j] - other.data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; ++k) {
                sum += data_[i][k] * other.data_[k][j];
            }
            result.data_[i][j] = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[i][j] = data_[i][j] * scalar;
        }
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    validateDimensions(other);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] += other.data_[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    validateDimensions(other);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] -= other.data_[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] *= scalar;
        }
    }
    return *this;
}

std::vector<double>& Matrix::operator[](size_t row) {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of range");
    }
    return data_[row];
}

const std::vector<double>& Matrix::operator[](size_t row) const {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of range");
    }
    return data_[row];
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[j][i] = data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square for inverse");
    }
    
    // Implement Gauss-Jordan elimination
    Matrix augmented(rows_, 2 * cols_);
    
    // Create augmented matrix [A|I]
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            augmented.data_[i][j] = data_[i][j];
            augmented.data_[i][j + cols_] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Gaussian elimination
    for (size_t i = 0; i < rows_; ++i) {
        double pivot = augmented.data_[i][i];
        if (std::abs(pivot) < 1e-10) {
            throw std::runtime_error("Matrix is singular");
        }
        
        // Scale pivot row
        for (size_t j = 0; j < 2 * cols_; ++j) {
            augmented.data_[i][j] /= pivot;
        }
        
        // Eliminate column
        for (size_t k = 0; k < rows_; ++k) {
            if (k != i) {
                double factor = augmented.data_[k][i];
                for (size_t j = 0; j < 2 * cols_; ++j) {
                    augmented.data_[k][j] -= factor * augmented.data_[i][j];
                }
            }
        }
    }
    
    // Extract inverse from augmented matrix
    Matrix inverse(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            inverse.data_[i][j] = augmented.data_[i][j + cols_];
        }
    }
    
    return inverse;
}

double Matrix::determinant() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square for determinant");
    }
    
    if (rows_ == 1) {
        return data_[0][0];
    }
    
    if (rows_ == 2) {
        return data_[0][0] * data_[1][1] - data_[0][1] * data_[1][0];
    }
    
    double det = 0.0;
    for (size_t j = 0; j < cols_; ++j) {
        Matrix minor(rows_ - 1, cols_ - 1);
        for (size_t i = 1; i < rows_; ++i) {
            size_t k = 0;
            for (size_t l = 0; l < cols_; ++l) {
                if (l != j) {
                    minor.data_[i-1][k] = data_[i][l];
                    ++k;
                }
            }
        }
        det += (j % 2 == 0 ? 1 : -1) * data_[0][j] * minor.determinant();
    }
    return det;
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size);
    for (size_t i = 0; i < size; ++i) {
        result.data_[i][i] = 1.0;
    }
    return result;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    Matrix result(rows, cols);
    for (auto& row : result.data_) {
        std::fill(row.begin(), row.end(), 1.0);
    }
    return result;
}

void Matrix::reshape(size_t rows, size_t cols) {
    if (rows * cols != rows_ * cols_) {
        throw std::invalid_argument("New dimensions must preserve total size");
    }
    
    std::vector<double> temp;
    temp.reserve(rows_ * cols_);
    
    for (const auto& row : data_) {
        temp.insert(temp.end(), row.begin(), row.end());
    }
    
    data_.resize(rows);
    for (auto& row : data_) {
        row.resize(cols);
    }
    
    size_t k = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data_[i][j] = temp[k++];
        }
    }
    
    rows_ = rows;
    cols_ = cols;
}

void Matrix::validateDimensions(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < matrix.rows_; ++i) {
        os << "[";
        for (size_t j = 0; j < matrix.cols_; ++j) {
            os << std::setw(8) << matrix.data_[i][j];
            if (j < matrix.cols_ - 1) os << ", ";
        }
        os << "]\n";
    }
    return os;
}

} // namespace utils
} // namespace ml