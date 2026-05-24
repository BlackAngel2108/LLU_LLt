#include "gtest/gtest.h"
#include "matrix.h"
#include <cmath>

#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

void test_rvv_intrinsics(void)
{
    int32_t input[4]  = {1, 2, 3, 4};
    int32_t output[4] = {0};

    size_t vl = __riscv_vsetvl_e32m1(4);

    vint32m1_t v = __riscv_vle32_v_i32m1(input, vl);
    v = __riscv_vadd_vx_i32m1(v, 10, vl);
    __riscv_vse32_v_i32m1(output, v, vl);

    printf("RVV intrinsics compiled and ran: %d %d %d %d\n",
           output[0], output[1], output[2], output[3]);
}


// Вспомогательная функция для сравнения матриц с допуском
void ASSERT_MATRICES_NEAR(const Matrix &a, const Matrix &b, double tolerance)
{
    ASSERT_EQ(a.getRows(), b.getRows());
    ASSERT_EQ(a.getCols(), b.getCols());
    for (int i = 0; i < a.getRows(); ++i)
    {
        for (int j = 0; j < a.getCols(); ++j)
        {
            ASSERT_NEAR(a(i, j), b(i, j), tolerance);
        }
    }
}



TEST(MatrixDecomposition, test_rvv)
{
    test_rvv_intrinsics();

    EXPECT_NE(2,3);
}

TEST(MatrixDecomposition, LUSimple)
{
    Matrix A(3, 3);
    A(0, 0) = 2;
    A(0, 1) = -1;
    A(0, 2) = -2;
    A(1, 0) = -4;
    A(1, 1) = 6;
    A(1, 2) = 3;
    A(2, 0) = -4;
    A(2, 1) = -2;
    A(2, 2) = 8;

    Matrix P(3, 3), L(3, 3), U(3, 3);
    A.lu_simple(P, L, U);
    Matrix left = P * A;
    Matrix right = L * U;

    ASSERT_MATRICES_NEAR(left, right, 1e-9);
}

TEST(MatrixDecomposition, LUBlocked)
{
    Matrix A(3, 3);
    A(0, 0) = 2;
    A(0, 1) = -1;
    A(0, 2) = -2;
    A(1, 0) = -4;
    A(1, 1) = 6;
    A(1, 2) = 3;
    A(2, 0) = -4;
    A(2, 1) = -2;
    A(2, 2) = 8;

    Matrix P(3, 3), L(3, 3), U(3, 3);
    A.lu_blocked(P, L, U, 128, 64);
    Matrix left = P * A;
    Matrix right = L * U;

    ASSERT_MATRICES_NEAR(left, right, 1e-9);
}

TEST(MatrixDecomposition, LUBlockedParallel)
{
    Matrix A(3, 3);
    A(0, 0) = 2;
    A(0, 1) = -1;
    A(0, 2) = -2;
    A(1, 0) = -4;
    A(1, 1) = 6;
    A(1, 2) = 3;
    A(2, 0) = -4;
    A(2, 1) = -2;
    A(2, 2) = 8;

    Matrix P(3, 3), L(3, 3), U(3, 3);
    A.lu_blocked_parallel(P, L, U, 128, 64);
    Matrix left = P * A;
    Matrix right = L * U;

    ASSERT_MATRICES_NEAR(left, right, 1e-9);
}

TEST(MatrixDecomposition, CholeskySimple)
{
    Matrix A(3, 3);
    A(0, 0) = 4;
    A(0, 1) = 12;
    A(0, 2) = -16;
    A(1, 0) = 12;
    A(1, 1) = 37;
    A(1, 2) = -43;
    A(2, 0) = -16;
    A(2, 1) = -43;
    A(2, 2) = 98;

    Matrix L(3, 3);                    // Создаём матрицу для результата
    A.cholesky_blocked_parallel(L, 2); // Передаём L и размер блока (2 по умолчанию)
    Matrix L_t = L.transpose();
    Matrix A_prime = L * L_t;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecomposition, CholeskyBlocked)
{
    Matrix A(3, 3);
    A(0, 0) = 4;
    A(0, 1) = 12;
    A(0, 2) = -16;
    A(1, 0) = 12;
    A(1, 1) = 37;
    A(1, 2) = -43;
    A(2, 0) = -16;
    A(2, 1) = -43;
    A(2, 2) = 98;

    Matrix L(3, 3);                    // Создаём матрицу для результата
    A.cholesky_blocked_parallel(L, 2); // Передаём L и размер блока (2 по умолчанию)
    Matrix L_t = L.transpose();
    Matrix A_prime = L * L_t;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecomposition, CholeskyBlockedParallel)
{
    Matrix A(3, 3);
    A(0, 0) = 4;
    A(0, 1) = 12;
    A(0, 2) = -16;
    A(1, 0) = 12;
    A(1, 1) = 37;
    A(1, 2) = -43;
    A(2, 0) = -16;
    A(2, 1) = -43;
    A(2, 2) = 98;

    Matrix L(3, 3);                    // Создаём матрицу для результата
    A.cholesky_blocked_parallel(L, 2); // Передаём L и размер блока (2 по умолчанию)

    Matrix L_t = L.transpose();
    Matrix A_prime = L * L_t;

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}


#ifdef USE_RVV

TEST(MatrixDecompositionRVV, LUBlockedParallelRVVM1) {
    Matrix A(3, 3);
    A(0, 0) = 2;  A(0, 1) = -1; A(0, 2) = -2;
    A(1, 0) = -4; A(1, 1) = 6;  A(1, 2) = 3;
    A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

    Matrix P(3, 3), L(3, 3), U(3, 3);
    A.lu_blocked_parallel_rvv_m1(P, L, U, 128, 64);

    Matrix left = P * A;
    Matrix right = L * U;

    ASSERT_MATRICES_NEAR(left, right, 1e-9);
}

TEST(MatrixDecompositionRVV, LUBlockedParallelRVVM2) {
    Matrix A(3, 3);
    A(0, 0) = 2;  A(0, 1) = -1; A(0, 2) = -2;
    A(1, 0) = -4; A(1, 1) = 6;  A(1, 2) = 3;
    A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

    Matrix P(3, 3), L(3, 3), U(3, 3);
    A.lu_blocked_parallel_rvv_m2(P, L, U, 128, 64);

    Matrix left = P * A;
    Matrix right = L * U;

    ASSERT_MATRICES_NEAR(left, right, 1e-9);
}

TEST(MatrixDecompositionRVV, LUBlockedParallelRVVM4) {
    Matrix A(3, 3);
    A(0, 0) = 2;  A(0, 1) = -1; A(0, 2) = -2;
    A(1, 0) = -4; A(1, 1) = 6;  A(1, 2) = 3;
    A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

    Matrix P(3, 3), L(3, 3), U(3, 3);
    A.lu_blocked_parallel_rvv_m4(P, L, U, 128, 64);

    Matrix left = P * A;
    Matrix right = L * U;

    ASSERT_MATRICES_NEAR(left, right, 1e-9);
}

TEST(MatrixDecompositionRVV, LUBlockedParallelRVVM8) {
    Matrix A(3, 3);
    A(0, 0) = 2;  A(0, 1) = -1; A(0, 2) = -2;
    A(1, 0) = -4; A(1, 1) = 6;  A(1, 2) = 3;
    A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

    Matrix P(3, 3), L(3, 3), U(3, 3);
    A.lu_blocked_parallel_rvv_m8(P, L, U, 128, 64);

    Matrix left = P * A;
    Matrix right = L * U;

    ASSERT_MATRICES_NEAR(left, right, 1e-9);
}

TEST(MatrixDecompositionRVV, CholeskyBlockedParallelRVVM1) {
    Matrix A(3, 3);
    A(0, 0) = 4;   A(0, 1) = 12;  A(0, 2) = -16;
    A(1, 0) = 12;  A(1, 1) = 37;  A(1, 2) = -43;
    A(2, 0) = -16; A(2, 1) = -43; A(2, 2) = 98;

    Matrix L(3, 3);
    A.cholesky_blocked_parallel_rvv_m1(L, 2);

    Matrix A_prime = L * L.transpose();

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecompositionRVV, CholeskyBlockedParallelRVVM2) {
    Matrix A(3, 3);
    A(0, 0) = 4;   A(0, 1) = 12;  A(0, 2) = -16;
    A(1, 0) = 12;  A(1, 1) = 37;  A(1, 2) = -43;
    A(2, 0) = -16; A(2, 1) = -43; A(2, 2) = 98;

    Matrix L(3, 3);
    A.cholesky_blocked_parallel_rvv_m2(L, 2);

    Matrix A_prime = L * L.transpose();

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecompositionRVV, CholeskyBlockedParallelRVVM4) {
    Matrix A(3, 3);
    A(0, 0) = 4;   A(0, 1) = 12;  A(0, 2) = -16;
    A(1, 0) = 12;  A(1, 1) = 37;  A(1, 2) = -43;
    A(2, 0) = -16; A(2, 1) = -43; A(2, 2) = 98;

    Matrix L(3, 3);
    A.cholesky_blocked_parallel_rvv_m4(L, 2);

    Matrix A_prime = L * L.transpose();

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

TEST(MatrixDecompositionRVV, CholeskyBlockedParallelRVVM8) {
    Matrix A(3, 3);
    A(0, 0) = 4;   A(0, 1) = 12;  A(0, 2) = -16;
    A(1, 0) = 12;  A(1, 1) = 37;  A(1, 2) = -43;
    A(2, 0) = -16; A(2, 1) = -43; A(2, 2) = 98;

    Matrix L(3, 3);
    A.cholesky_blocked_parallel_rvv_m8(L, 2);

    Matrix A_prime = L * L.transpose();

    ASSERT_MATRICES_NEAR(A, A_prime, 1e-9);
}

#endif

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}