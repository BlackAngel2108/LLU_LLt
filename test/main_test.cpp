#include "gtest/gtest.h"
#include "matrix.h"
#include <cmath>

// Вспомогательная функция для сравнения матриц с допуском
void ASSERT_MATRICES_NEAR(const Matrix& a, const Matrix& b, double tolerance) {
    ASSERT_EQ(a.getRows(), b.getRows());
    ASSERT_EQ(a.getCols(), b.getCols());
    for (int i = 0; i < a.getRows(); ++i) {
        for (int j = 0; j < a.getCols(); ++j) {
            ASSERT_NEAR(a(i, j), b(i, j), tolerance);
        }
    }
}

TEST(MatrixDecomposition, LUSimple) {
    Matrix A(3, 3);
    A(0, 0) = 2; A(0, 1) = -1; A(0, 2) = -2;
    A(1, 0) = -4; A(1, 1) = 6; A(1, 2) = 3;
    A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

    auto lu_pair = A.lu_simple();
    Matrix L = lu_pair.first;
    Matrix U = lu_pair.second;
    Matrix A_prime = L * U;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecomposition, LUBlocked) {
    Matrix A(3, 3);
    A(0, 0) = 2; A(0, 1) = -1; A(0, 2) = -2;
    A(1, 0) = -4; A(1, 1) = 6; A(1, 2) = 3;
    A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

    auto lu_pair = A.lu_blocked(2);
    Matrix L = lu_pair.first;
    Matrix U = lu_pair.second;
    Matrix A_prime = L * U;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecomposition, LUBlockedParallel) {
    Matrix A(3, 3);
    A(0, 0) = 2; A(0, 1) = -1; A(0, 2) = -2;
    A(1, 0) = -4; A(1, 1) = 6; A(1, 2) = 3;
    A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

    auto lu_pair = A.lu_blocked_parallel(2);
    Matrix L = lu_pair.first;
    Matrix U = lu_pair.second;
    Matrix A_prime = L * U;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecomposition, CholeskySimple) {
    Matrix A(3, 3);
    A(0, 0) = 4;  A(0, 1) = 12; A(0, 2) = -16;
    A(1, 0) = 12; A(1, 1) = 37; A(1, 2) = -43;
    A(2, 0) = -16; A(2, 1) = -43; A(2, 2) = 98;

    Matrix L = A.cholesky();
    Matrix L_t = L.transpose();
    Matrix A_prime = L * L_t;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecomposition, CholeskyBlocked) {
    Matrix A(3, 3);
    A(0, 0) = 4;  A(0, 1) = 12; A(0, 2) = -16;
    A(1, 0) = 12; A(1, 1) = 37; A(1, 2) = -43;
    A(2, 0) = -16; A(2, 1) = -43; A(2, 2) = 98;

    Matrix L = A.cholesky_blocked(2);
    Matrix L_t = L.transpose();
    Matrix A_prime = L * L_t;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecomposition, CholeskyBlockedParallel) {
    Matrix A(3, 3);
    A(0, 0) = 4;  A(0, 1) = 12; A(0, 2) = -16;
    A(1, 0) = 12; A(1, 1) = 37; A(1, 2) = -43;
    A(2, 0) = -16; A(2, 1) = -43; A(2, 2) = 98;

    Matrix L = A.cholesky_blocked_parallel(2);
    Matrix L_t = L.transpose();
    Matrix A_prime = L * L_t;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}