#include <array>
#include <boost/optional/optional.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include <cblas.h>

class Matrix {
    std::size_t _rows;
    std::size_t _cols;
public:
    float *data;

    Matrix(const std::size_t numRows, const std::size_t numCols, const boost::optional<float> initValue = boost::none) : _rows(numRows), _cols(numCols) {
        data = new float[_rows * _cols];
        if (initValue) {
            for (std::size_t row = 0; row < numRows; ++row) {
            for (std::size_t col = 0; col < numCols; ++col) {
                data[row + col] = initValue.get();
            }
            }
        } else {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<float> dist(0, 1000);

            for (std::size_t row = 0; row < numRows; ++row) {
            for (std::size_t col = 0; col < numCols; ++col) {
                data[row + col] = dist(generator);
            }
            }
        }
    }

    ~Matrix() {
        delete[] data;
        data = nullptr;
    }

    decltype(_rows) rows() const {
        return _rows;
    }
    decltype(_cols) cols() const {
        return _cols;
    }

    static void gemm_naive(const Matrix & A, const Matrix & B, Matrix & C) {
        for (std::size_t row = 0; row < A.rows(); ++row) {
        for (std::size_t col = 0; col < A.cols(); ++col) {
        for (std::size_t i = 0; i < A.rows(); ++i) {
            C.data[row + col] = C.data[row + col] + A.data[row+i] * B.data[i+col];
        }
        }
        }
    }
};

std::ostream& operator<<(std::ostream & stream, const Matrix & m) {
    for (std::size_t row = 0; row < m.rows(); ++row) {
        for (std::size_t col = 0; col < m.cols(); ++col) {
            stream << m.data[row + col] << " ";
        }
        stream << "\n";
    }
    return stream;
}

std::ostream& aligned(const float number, const int width, const int precision) {
    return std::cout << std::setw(width) << std::setiosflags(std::ios::fixed) << std::setprecision(precision) << std::right << number;
}

int main(int, char *[]) {
    const std::size_t samplesPerDim = 10;
    const std::size_t numDims = 1;
    const std::size_t stepSize = 100;
    std::array<std::pair<float, float>, numDims> results;

    for (std::size_t i = 0; i < numDims; ++i) {
        const std::size_t dim = (i+1)*stepSize;
        long avgNaiveGemm = 0;
        long avgCblasGemm = 0;
        for (uint sample = 0; sample < samplesPerDim; ++sample) {
            const Matrix A(dim, dim);
            const Matrix B(dim, dim);
            Matrix C(dim, dim);

            auto startNaive = std::chrono::steady_clock::now();
            Matrix::gemm_naive(A, B, C);
            avgNaiveGemm += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startNaive).count();

            const int intDim = static_cast<int>(dim);
            auto startCblas = std::chrono::steady_clock::now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        intDim, intDim, intDim,
                        1, A.data, intDim, B.data, intDim, 1, C.data, intDim);
            avgCblasGemm += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startCblas).count();
        }
        results[i] = std::make_pair(static_cast<float>(avgNaiveGemm)/samplesPerDim, static_cast<float>(avgCblasGemm)/samplesPerDim);
        std::cout << "finished dimension " << (i+1)*stepSize << std::endl;
    }

    std::cout << "Dimensions: " << numDims << ", step size: " << stepSize << ", samples per dimension: " << samplesPerDim << std::endl;
    for (std::size_t i = 0; i < numDims; ++i) {
        auto naive = results[i].first;
        auto cblas = results[i].second;
        std::cout << "dim: "; aligned((i+1)*stepSize, 4, 0) << ". Naive gemm: "; aligned(naive/std::pow(10.f, 6.f), 8, 3) << "ms ("; aligned(naive/std::pow(10.f, 9.f), 5, 3);
        std::cout << "s), cblas gemm: "; aligned(cblas/std::pow(10.f, 6.f), 8, 3) << "ms ("; aligned(cblas/std::pow(10.f, 9.f), 5, 3) << "ss).";
        if (cblas > 0) {
            std::cout << " naive/cblas: "; aligned(static_cast<float>(naive)/cblas, 5, 2);
        }
        std::cout << std::endl;
    }
}
