#include "matrix.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath> // For std::sqrt

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Размеры матрицы должны быть положительными.");
    }
}

Matrix::Matrix(const Matrix &other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

Matrix &Matrix::operator=(const Matrix &other)
{
    if (this == &other)
    {
        return *this;
    }
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
    return *this;
}

double &Matrix::operator()(int row, int col)
{
    if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
    {
        throw std::out_of_range("Индекс за пределами матрицы.");
    }
    return data_[row * cols_ + col];
}

const double &Matrix::operator()(int row, int col) const
{
    if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
    {
        throw std::out_of_range("Индекс за пределами матрицы.");
    }
    return data_[row * cols_ + col];
}

int Matrix::getRows() const
{
    return rows_;
}

int Matrix::getCols() const
{
    return cols_;
}

void Matrix::fillRandom()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 100.0);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        data_[i] = dis(gen);
    }
}

void Matrix::print() const
{
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Реализация трех версий LU-разложения

// 1. Простой алгоритм LU-разложения
std::pair<Matrix, Matrix> Matrix::lu_simple() const
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");
    }

    int n = rows_;
    Matrix L(n, n);
    Matrix U = *this;

    for (int i = 0; i < n; ++i)
    {
        L(i, i) = 1.0;
        for (int j = i + 1; j < n; ++j)
        {
            double factor = U(j, i) / U(i, i);
            L(j, i) = factor;
            for (int k = i; k < n; ++k)
            {
                U(j, k) -= factor * U(i, k);
            }
        }
    }
    return {L, U};
}

// 2. Блочный алгоритм LU-разложения
std::pair<Matrix, Matrix> Matrix::lu_blocked(int block_size) const
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");
    }

    int n = rows_;
    Matrix LU_combined = *this; // Используем комбинированное хранение L и U [3]

    for (int k = 0; k < n; k += block_size)
    {
        int kb = std::min(n, k + block_size); // Граница текущего блока

        // 1. Обработка панели (аналогично cholesky_blocked [4])
        // Выполняем обычное LU внутри текущего столбца блоков
        for (int i = k; i < kb; ++i)
        {
            if (LU_combined(i, i) == 0)
                throw std::runtime_error("Деление на ноль.");

            for (int j = i + 1; j < n; ++j)
            {
                LU_combined(j, i) /= LU_combined(i, i); // Вычисляем множители L
                double factor = LU_combined(j, i);

                // Обновляем только элементы внутри текущей панели (до kb)
                for (int l = i + 1; l < kb; ++l)
                {
                    LU_combined(j, l) -= factor * LU_combined(i, l);
                }
            }
        }

        // 2. Обновление текущей строки блоков (U-часть справа от диагонального блока)
        for (int i = k; i < kb; ++i)
        {
            for (int j = kb; j < n; ++j)
            {
                for (int l = k; l < i; ++l)
                {
                    LU_combined(i, j) -= LU_combined(i, l) * LU_combined(l, j);
                }
            }
        }

        // 3. Обновление оставшейся подматрицы (Trailing Submatrix Update) [5]

        for (int i = kb; i < n; ++i)
        {
            for (int j = kb; j < n; ++j)
            {
                double sum = 0.0;
                for (int l = k; l < kb; ++l)
                {
                    sum += LU_combined(i, l) * LU_combined(l, j);
                }
                LU_combined(i, j) -= sum;
            }
        }
    }

    // Извлечение матриц L и U (сохраняем вашу оригинальную логику из источника [3, 6])
    Matrix L(n, n);
    Matrix U(n, n);
    for (int i = 0; i < n; ++i)
    {
        L(i, i) = 1.0;
        for (int j = 0; j < i; ++j)
        {
            L(i, j) = LU_combined(i, j);
        }
        for (int j = i; j < n; ++j)
        {
            U(i, j) = LU_combined(i, j);
        }
    }
    return {L, U};
}

// 3. Блочный параллельный алгоритм LU-разложения с OpenMP
std::pair<Matrix, Matrix> Matrix::lu_blocked_parallel(int block_size) const
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");
    }

    int n = rows_;
    Matrix LU_combined = *this; // Используем комбинированное хранение L и U [3]

    for (int k = 0; k < n; k += block_size)
    {
        int kb = std::min(n, k + block_size); // Граница текущего блока

        // 1. Обработка панели (аналогично cholesky_blocked [4])
        // Выполняем обычное LU внутри текущего столбца блоков
        for (int i = k; i < kb; ++i)
        {
            if (LU_combined(i, i) == 0)
                throw std::runtime_error("Деление на ноль.");

            for (int j = i + 1; j < n; ++j)
            {
                LU_combined(j, i) /= LU_combined(i, i); // Вычисляем множители L
                double factor = LU_combined(j, i);

                // Обновляем только элементы внутри текущей панели (до kb)
                for (int l = i + 1; l < kb; ++l)
                {
                    LU_combined(j, l) -= factor * LU_combined(i, l);
                }
            }
        }

        // 2. Обновление текущей строки блоков (U-часть справа от диагонального блока)
        for (int i = k; i < kb; ++i)
        {
            for (int j = kb; j < n; ++j)
            {
                for (int l = k; l < i; ++l)
                {
                    LU_combined(i, j) -= LU_combined(i, l) * LU_combined(l, j);
                }
            }
        }

// 3. Обновление оставшейся подматрицы (Trailing Submatrix Update) [5]
// Это самая вычислительно затратная часть, которую мы распараллеливаем
#pragma omp parallel for collapse(2) if (n - kb > block_size)
        for (int i = kb; i < n; ++i)
        {
            for (int j = kb; j < n; ++j)
            {
                double sum = 0.0;
                for (int l = k; l < kb; ++l)
                {
                    sum += LU_combined(i, l) * LU_combined(l, j);
                }
                LU_combined(i, j) -= sum;
            }
        }
    }

    // Извлечение матриц L и U (сохраняем вашу оригинальную логику из источника [3, 6])
    Matrix L(n, n);
    Matrix U(n, n);
    for (int i = 0; i < n; ++i)
    {
        L(i, i) = 1.0;
        for (int j = 0; j < i; ++j)
        {
            L(i, j) = LU_combined(i, j);
        }
        for (int j = i; j < n; ++j)
        {
            U(i, j) = LU_combined(i, j);
        }
    }
    return {L, U};
}

// 4. Разложение Холецкого (LLt)
Matrix Matrix::cholesky() const
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }
    int n = rows_;
    Matrix L(n, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
            {
                sum += L(i, k) * L(j, k);
            }

            if (i == j)
            {
                double diagonal_val = data_[i * n + i] - sum;
                if (diagonal_val <= 0)
                {
                    throw std::runtime_error("Матрица не является положительно-определенной.");
                }
                L(i, i) = std::sqrt(diagonal_val);
            }
            else
            {
                if (L(j, j) == 0)
                {
                    throw std::runtime_error("Деление на ноль в разложении Холецкого.");
                }
                L(i, j) = (data_[i * n + j] - sum) / L(j, j);
            }
        }
    }
    return L;
}

// 1. Простой параллельный алгоритм LU-разложения с OpenMP
std::pair<Matrix, Matrix> Matrix::lu_simple_parallel()
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");
    }

    int n = rows_;
    Matrix L(n, n);
    Matrix U = *this;

    for (int i = 0; i < n; ++i)
    {
        L(i, i) = 1.0;
        if (U(i, i) == 0)
        { // Check for pivot zero
            throw std::runtime_error("Нулевой элемент на диагонали в LU-разложении.");
        }

#pragma omp parallel for
        for (int j = i + 1; j < n; ++j)
        {
            double factor = U(j, i) / U(i, i);
            L(j, i) = factor;
            for (int k = i; k < n; ++k)
            {
                U(j, k) -= factor * U(i, k);
            }
        }
    }
    return {L, U};
}

// 4. Параллельный алгоритм разложения Холецкого (LLt) с OpenMP
Matrix Matrix::cholesky_simple_parallel()
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }
    int n = rows_;
    Matrix L(n, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int k = 0; k < j; k++)
            {
                sum += L(i, k) * L(j, k);
            }

            if (i == j)
            {
                double diagonal_val = data_[i * n + i] - sum;
                if (diagonal_val <= 0)
                {
                    throw std::runtime_error("Матрица не является положительно-определенной.");
                }
                L(i, i) = std::sqrt(diagonal_val);
            }
            else
            {
                if (L(j, j) == 0)
                {
                    throw std::runtime_error("Деление на ноль в разложении Холецкого.");
                }
                L(i, j) = (data_[i * n + j] - sum) / L(j, j);
            }
        }
    }
    return L;
}

// Умножение матриц
Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols_ != other.rows_)
    {
        throw std::invalid_argument("Несоответствие размеров матриц для умножения.");
    }
    Matrix result(rows_, other.cols_);
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < other.cols_; ++j)
        {
            for (int k = 0; k < cols_; ++k)
            {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return result;
}

// Транспонирование матрицы
Matrix Matrix::transpose() const
{
    Matrix result(cols_, rows_);
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Блочный алгоритм разложения Холецкого (из otchet.txt)
Matrix Matrix::cholesky_blocked(int block_size) const
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }
    int n = rows_;
    Matrix L_result = *this; // A acts as L in-place

    for (int k = 0; k < n; k += block_size)
    {
        int kb = std::min(n, k + block_size);
        // 1. Process Panel (Diagonal block + column below)
        for (int j = k; j < kb; ++j)
        {
            double d = L_result(j, j);
            for (int l = k; l < j; ++l)
            {
                double val = L_result(j, l);
                d -= val * val;
            }
            d = std::sqrt(d > 0 ? d : 1e-9);
            L_result(j, j) = d;
            double invDiag = 1.0 / d;
            for (int i = j + 1; i < n; ++i)
            {
                double s = L_result(i, j);
                for (int l = k; l < j; ++l)
                    s -= L_result(i, l) * L_result(j, l);
                L_result(i, j) = s * invDiag;
            }
        }
        // 2. Update Trailing Submatrix (Sequential GEMM-like)
        for (int i = kb; i < n; ++i)
        {
            for (int j = kb; j <= i; ++j)
            {
                double sum = 0.0;
                for (int l = k; l < kb; ++l)
                {
                    sum += L_result(i, l) * L_result(j, l);
                }
                L_result(i, j) -= sum;
            }
        }
    }
    // Set upper triangle to 0
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            L_result(i, j) = 0.0;
        }
    }
    return L_result;
}

// Блочный параллельный алгоритм разложения Холецкого (из otchet.txt)
Matrix Matrix::cholesky_blocked_parallel(int block_size) const
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }
    int n = rows_;
    Matrix L_result = *this; // A acts as L in-place

    for (int k = 0; k < n; k += block_size)
    {
        int kb = std::min(n, k + block_size);
        // 1. Process Panel (Diagonal block + column below)
        for (int j = k; j < kb; ++j)
        {
            double d = L_result(j, j);
            for (int l = k; l < j; ++l)
            {
                double val = L_result(j, l);
                d -= val * val;
            }
            d = std::sqrt(d > 0 ? d : 1e-9);
            L_result(j, j) = d;
            double invDiag = 1.0 / d;
            for (int i = j + 1; i < n; ++i)
            {
                double s = L_result(i, j);
                for (int l = k; l < j; ++l)
                    s -= L_result(i, l) * L_result(j, l);
                L_result(i, j) = s * invDiag;
            }
        }
// 2. Update Trailing Submatrix (Parallel GEMM-like)
#pragma omp parallel for schedule(guided)
        for (int i = kb; i < n; ++i)
        {
            for (int j = kb; j <= i; ++j)
            {
                double sum = 0.0;
                for (int l = k; l < kb; ++l)
                {
                    sum += L_result(i, l) * L_result(j, l);
                }
                L_result(i, j) -= sum;
            }
        }
    }
    // Set upper triangle to 0
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            L_result(i, j) = 0.0;
        }
    }
    return L_result;
}