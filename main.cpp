#include <array>
#include <boost/optional/optional.hpp>
#include <iostream>
#include <random>
#include <sstream>

template<std::size_t numRows, std::size_t numCols>
class Matrix {
public:
    std::array<std::array<int, numCols>, numRows> data;
    Matrix(const boost::optional<int> initValue = boost::none) {
        if (initValue) {
            for (std::size_t row = 0; row < numRows; ++row) {
            for (std::size_t col = 0; col < numCols; ++col) {
                data[row][col] = initValue.get();
            }
            }
        } else {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_int_distribution<> dist(0, 1000);

            for (std::size_t row = 0; row < numRows; ++row) {
            for (std::size_t col = 0; col < numCols; ++col) {
                data[row][col] = dist(generator);
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
            sum.data[row][col] = data[row][col] + other.data[row][col];
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
            product.data[row][col] += data[row][i] * other.data[i][col];
        }
        }
        }
        return product;
    }
};
template<std::size_t numRows, std::size_t numCols>
std::ostream& operator<<(std::ostream &stream, const Matrix<numRows, numCols> & m) {
    for (std::size_t row = 0; row < m.rows(); ++row) {
        for (std::size_t col = 0; col < m.cols(); ++col) {
            stream << m.data[row][col] << " ";
        }
        stream << "\n";
    }
    return stream;
}

int main(int, char *[]) {
    const Matrix<3, 3> A;
    const Matrix<3, 3> B;
    std::cout << "A:\n" << A;
    std::cout << "B:\n" << B;
    std::cout << "A + B:\n" << A + B;
    std::cout << "A * B:\n" << A * B;

}
