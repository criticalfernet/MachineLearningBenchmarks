#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;

Matrix multiply(const Matrix& A, const Matrix& B) {
    return A * B;
}