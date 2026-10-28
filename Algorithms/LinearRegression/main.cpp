#include <Eigen/Dense>
#include <Eigen/Cholesky>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

void linear_regression(const Matrix& X, const Vector& y, Vector& C) {
    Matrix G = X.transpose() * X;
    Vector b = X.transpose() * y;

    C.noalias() = G.llt().solve(b);
}