#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

namespace ml {
namespace utils {

class Matrix {
public:
    /**
     * @brief Construct a new Matrix object
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(size_t rows = 0, size_t cols = 0);
    
    /**
     * @brief Construct a Matrix from vector of vectors
     * @param data Input data
     */
    explicit Matrix(const std::vector<std::vector<double>>& data);

    // Rule of five
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix() = default;

    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

    // Access operators
    std::vector<double>& operator[](size_t row);
    const std::vector<double>& operator[](size_t row) const;

    // Matrix operations
    Matrix transpose() const;
    Matrix inverse() const;
    double determinant() const;
    
    // Static methods
    static Matrix identity(size_t size);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);

    // Utility methods
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    void reshape(size_t rows, size_t cols);
    
    // I/O operations
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

private:
    size_t rows_;
    size_t cols_;
    std::vector<std::vector<double>> data_;

    void validateDimensions(const Matrix& other) const;
};

} // namespace utils
} // namespace ml