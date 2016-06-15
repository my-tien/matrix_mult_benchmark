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
    const std::size_t samplesPerDim = 100;
    const std::size_t numDims = 10;
    const std::size_t stepSize = 100;
    std::array<std::array<float, 3>, numDims> results;

    for (std::size_t i = 0; i < numDims; ++i) {
        const std::size_t dim = (i+1)*stepSize;
        long avgNaiveGemm = 0;
        long avgCblasGemmSingle = 0;
        long avgCblasGemmMulti = 0;
        const Matrix A(dim, dim);
        std::cout << "  finished building matrix A" << std::endl;
        const Matrix B(dim, dim);
        std::cout << "  finished building matrix B" << std::endl;
        Matrix C(dim, dim);
        std::cout << "  finished building matrix C" << std::endl;
        for (uint sample = 0; sample < samplesPerDim; ++sample) {
            std::cout << "Dimension " << dim << ", sample " << sample + 1 << std::endl;
            const int intDim = static_cast<int>(dim);

            auto startNaive = std::chrono::steady_clock::now();
            Matrix::gemm_naive(A, B, C);
            avgNaiveGemm += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startNaive).count();
            std::cout << "  finished naive gemm" << std::endl;

            openblas_set_num_threads(1);
            auto startCblas = std::chrono::steady_clock::now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        intDim, intDim, intDim,
                        1, A.data, intDim, B.data, intDim, 1, C.data, intDim);
            avgCblasGemmSingle += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startCblas).count();
            std::cout << "  finished cblas gemm single-core." << std::endl;

            openblas_set_num_threads(4);
            auto startCblasMulti = std::chrono::steady_clock::now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        intDim, intDim, intDim,
                        1, A.data, intDim, B.data, intDim, 1, C.data, intDim);
            std::cout << "  finished cblas gemm multi-core (4 threads)" << std::endl;
            avgCblasGemmMulti += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startCblasMulti).count();
        }
        results[i] = {static_cast<float>(avgNaiveGemm)/samplesPerDim, static_cast<float>(avgCblasGemmSingle)/samplesPerDim, static_cast<float>(avgCblasGemmMulti)/samplesPerDim};
        std::cout << "finished dimension " << dim << std::endl << std::endl;
    }

    std::cout << "Dimensions: " << numDims << ", step size: " << stepSize << ", samples per dimension: " << samplesPerDim << std::endl;
    for (std::size_t i = 0; i < numDims; ++i) {
        auto naive = results[i][0];
        auto cblasSingle = results[i][1];
        auto cblasMulti = results[i][2];
        std::cout << "dim "; aligned((i+1)*stepSize, 4, 0) << ":" << std::endl;
        std::cout << "Naive: "; aligned(naive/std::pow(10.f, 6.f), 8, 3) << "ms ("; aligned(naive/std::pow(10.f, 9.f), 5, 3);
        std::cout << "s), cblas single-core: "; aligned(cblasSingle/std::pow(10.f, 6.f), 8, 3) << "ms ("; aligned(cblasSingle/std::pow(10.f, 9.f), 5, 3) << "s).";
        std::cout << "s), cblas multi-core: "; aligned(cblasMulti/std::pow(10.f, 6.f), 8, 3) << "ms ("; aligned(cblasMulti/std::pow(10.f, 9.f), 5, 3) << "s)." << std::endl;
        if (cblasSingle > 0) {
            std::cout << "Ratio naive/cblas-single-core: "; aligned(static_cast<float>(naive)/cblasSingle, 5, 2) << std::endl;
            if (cblasMulti > 0) {
                std::cout << "Ratio naive/cblas-multi-core: "; aligned(static_cast<float>(naive)/cblasMulti, 5, 2) << ", cblas-single-core/cblas-multi-core: "; aligned(static_cast<float>(cblasSingle)/cblasMulti, 5, 2) << std::endl;
            }
        }
        std::cout << std::endl;
    }
}
