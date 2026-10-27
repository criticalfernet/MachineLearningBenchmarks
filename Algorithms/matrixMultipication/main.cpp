#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;

void multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    C.noalias() = A * B;
}