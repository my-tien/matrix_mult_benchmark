#include <array>
#include <boost/optional/optional.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>

#include <cblas.h>

template<std::size_t numRows, std::size_t numCols>
class Matrix {
public:
    float data[numRows*numCols];
    Matrix(const boost::optional<float> initValue = boost::none) {
        if (initValue) {
            for (std::size_t row = 0; row < numRows; ++row) {
            for (std::size_t col = 0; col < numCols; ++col) {
                data[row+col] = initValue.get();
            }
            }
        } else {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<float> dist(0, 1000);

            for (std::size_t row = 0; row < numRows; ++row) {
            for (std::size_t col = 0; col < numCols; ++col) {
                data[row+col] = dist(generator);
            }
            }
        }
    }

    decltype(numRows) rows() const {
        return numRows;
    }
    decltype(numCols) cols() const {
        return numCols;
    }

    Matrix operator+(const Matrix & other) const {
        if (other.rows() != numRows || other.cols() != numCols) {
            std::stringstream error;
            error << "Mismatching matrix dimensions, this: (" << numRows << "×" << numCols << "), other: (" << other.rows() << "×" << other.cols() << ")";
            throw std::runtime_error(error.str());
        }
        Matrix<numRows, numCols> sum(0);
        for (std::size_t row = 0; row < numRows; ++row) {
        for (std::size_t col = 0; col < numCols; ++col) {
            sum.data[row+col] = data[row+col] + other.data[row+col];
        }
        }
        return sum;
    }

    Matrix operator*(const Matrix & other) const {
        if (other.rows() != numCols || other.cols() != numRows) {
            std::stringstream error;
            error << "Mismatching matrix dimensions, this: (" << numRows << "×" << numCols << "), other: (" << other.rows() << "×" << other.cols() << ")";
            throw std::runtime_error(error.str());
        }
        Matrix<numRows, numCols> product(0);
        for (std::size_t row = 0; row < numRows; ++row) {
        for (std::size_t col = 0; col < numCols; ++col) {
        for (std::size_t i = 0; i < numRows; ++i) {
            product.data[row+col] += data[row+i] * other.data[i+col];
        }
        }
        }
        return product;
    }

    static void gemm_naive(const Matrix & A, const Matrix & B, Matrix & C) {
        if (A.rows() != B.cols() || A.cols() != B.rows()) {
            std::stringstream error;
            error << "Mismatching matrix dimensions, A(" << A.rows() << "×" << A.cols() << "), B(" << B.rows() << "×" << B.cols() << "), C(" << C.rows() << "×" << C.cols() << ")";
            throw std::runtime_error(error.str());
        }
        for (std::size_t row = 0; row < numRows; ++row) {
        for (std::size_t col = 0; col < numCols; ++col) {
        for (std::size_t i = 0; i < numRows; ++i) {
            C.data[row+col] += A.data[row+i] * B.data[i+col];
        }
        }
        }
    }
};
template<std::size_t numRows, std::size_t numCols>
std::ostream& operator<<(std::ostream &stream, const Matrix<numRows, numCols> & m) {
    for (std::size_t row = 0; row < m.rows(); ++row) {
        for (std::size_t col = 0; col < m.cols(); ++col) {
            stream << m.data[row+col] << " ";
        }
        stream << "\n";
    }
    return stream;
}

int main(int, char *[]) {
    const std::size_t dim = 1000;
    const auto A = new Matrix<dim, dim>();
    const auto B = new Matrix<dim, dim>();
    auto C = new Matrix<dim, dim>();
    auto start1 = std::chrono::steady_clock::now();
    Matrix<dim, dim>::gemm_naive(*A, *B, *C);
    auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start1).count();
    auto start2 = std::chrono::steady_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, dim, dim,
                1, A->data, dim, B->data, dim, 1, C->data, dim);
    auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start2).count();
    std::cout << "naive gemm: " << elapsed1 << "ms (" << elapsed1/1000. << "s), cblas gemm: " << elapsed2 << "ms (" << elapsed2/1000. << "s). " << std::endl;
}
