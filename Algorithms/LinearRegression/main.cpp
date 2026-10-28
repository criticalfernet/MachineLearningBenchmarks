#include <Eigen/Dense>
#include <Eigen/Cholesky>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Vector linear_regression(const Matrix& X, const Vector& y) {
    Matrix G = X.transpose() * X;
    Vector b = X.transpose() * y;

    return G.llt().solve(b);
}