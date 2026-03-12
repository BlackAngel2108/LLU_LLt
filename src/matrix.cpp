#include "matrix.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <numeric>   // Для std::iota
#include <cstring>   // Для std::memcpy
#include <algorithm> // для std::fill, std::copy
#include <immintrin.h>
#include <tuple>

// Вспомогательные функции для выровненной памяти
static double *aligned_alloc(size_t size)
{
    void *ptr = nullptr;
    // Выравнивание на 64 байта (для AVX-512)
    if (posix_memalign(&ptr, 64, size * sizeof(double)) != 0)
    {
        throw std::bad_alloc();
    }
    return static_cast<double *>(ptr);
}

static void aligned_free(double *ptr)
{
    free(ptr); // posix_memalign использует обычный free
}

// Конструктор
Matrix::Matrix(int rows, int cols)
    : rows_(rows), cols_(cols),
      data_(nullptr), data_size_(rows * cols),
      owns_data_(true)
{
    if (rows <= 0 || cols <= 0)
    {
        throw std::invalid_argument("Размеры матрицы должны быть положительными.");
    }
    data_ = aligned_alloc(data_size_);
    std::fill(data_, data_ + data_size_, 0.0);
}

// Конструктор копирования
Matrix::Matrix(const Matrix &other)
    : rows_(other.rows_), cols_(other.cols_),
      data_size_(other.data_size_), owns_data_(true)
{
    data_ = aligned_alloc(data_size_);
    std::memcpy(data_, other.data_, data_size_ * sizeof(double));
}

// Конструктор перемещения
Matrix::Matrix(Matrix &&other) noexcept
    : rows_(other.rows_), cols_(other.cols_),
      data_(other.data_), data_size_(other.data_size_),
      owns_data_(other.owns_data_)
{
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_size_ = 0;
    other.owns_data_ = false;
}

// Оператор присваивания
Matrix &Matrix::operator=(const Matrix &other)
{
    if (this == &other)
        return *this;

    // Освобождаем старые данные
    if (owns_data_ && data_)
    {
        aligned_free(data_);
    }

    rows_ = other.rows_;
    cols_ = other.cols_;
    data_size_ = other.data_size_;
    owns_data_ = true;

    data_ = aligned_alloc(data_size_);
    std::memcpy(data_, other.data_, data_size_ * sizeof(double));

    return *this;
}

// Оператор перемещения
Matrix &Matrix::operator=(Matrix &&other) noexcept
{
    if (this == &other)
        return *this;

    // Освобождаем старые данные
    if (owns_data_ && data_)
    {
        aligned_free(data_);
    }

    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
    data_size_ = other.data_size_;
    owns_data_ = other.owns_data_;

    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_size_ = 0;
    other.owns_data_ = false;

    return *this;
}

// Деструктор
Matrix::~Matrix()
{
    if (owns_data_ && data_)
    {
        aligned_free(data_);
    }
}

const double &Matrix::operator()(int row, int col) const
{
    if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
    {
        throw std::out_of_range("Индекс за пределами матрицы.");
    }
    return data_[row * cols_ + col];
}

double &Matrix::operator()(int row, int col)
{
    if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
    {
        throw std::out_of_range("Индекс за пределами матрицы.");
    }
    return data_[row * cols_ + col]; // прямой доступ, уже выровнено
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
    // Один генератор на поток (но без OpenMP это просто один)
    static thread_local std::mt19937 gen(std::random_device{}());

    std::uniform_real_distribution<double> dis(1.0, 100.0);

    const size_t total = static_cast<size_t>(rows_) * cols_;

    for (size_t i = 0; i < total; ++i)
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
// Подсказка компилятору о выравнивании
#ifdef __GNUC__
#define ASSUME_ALIGNED(ptr) __builtin_assume_aligned((ptr), 64)
#else
#define ASSUME_ALIGNED(ptr) (ptr)
#endif

std::tuple<Matrix, Matrix, Matrix>
Matrix::lu_simple() const
{
    if (rows_ != cols_)
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");

    const int n = rows_;

    Matrix A = *this; // in-place факторизация
    double *a = A.data();
    const int ld = cols_;

    auto at = [&](int r, int c) -> double &
    {
        return a[(size_t)r * ld + c];
    };

    // Pivot vector (как в LAPACK, 0-based)
    std::vector<int> ipiv(n);
    for (int i = 0; i < n; ++i)
        ipiv[i] = i;

    // ===== Классический GEPP =====
    for (int k = 0; k < n; ++k)
    {
        // ---- 1) Поиск главного элемента ----
        int p = k;
        double maxv = std::abs(at(k, k));

        for (int i = k + 1; i < n; ++i)
        {
            double v = std::abs(at(i, k));
            if (v > maxv)
            {
                maxv = v;
                p = i;
            }
        }

        if (maxv == 0.0)
            throw std::runtime_error("Матрица вырождена.");

        ipiv[k] = p;

        // ---- 2) Перестановка строк ----
        if (p != k)
        {
            for (int j = 0; j < n; ++j)
                std::swap(at(k, j), at(p, j));
        }

        // ---- 3) Вычисление множителей L ----
        double inv_piv = 1.0 / at(k, k);
        for (int i = k + 1; i < n; ++i)
            at(i, k) *= inv_piv;

        // ---- 4) Обновление хвоста ----
        double *Ak = a + (size_t)k * ld;

        for (int i = k + 1; i < n; ++i)
        {
            double *Ai = a + (size_t)i * ld;
            double lik = Ai[k];

// векторизуем по строке
#pragma omp simd
            for (int j = k + 1; j < n; ++j)
                Ai[j] -= lik * Ak[j];
        }
    }

    // ===== Извлечение L и U =====
    Matrix L(n, n);
    Matrix U(n, n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i > j)
            {
                L(i, j) = at(i, j);
                U(i, j) = 0.0;
            }
            else if (i == j)
            {
                L(i, j) = 1.0;
                U(i, j) = at(i, j);
            }
            else
            {
                L(i, j) = 0.0;
                U(i, j) = at(i, j);
            }
        }
    }

    // ===== Построение матрицы перестановок P =====
    Matrix P(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            P(i, j) = 0.0;

    // Строим итоговую перестановку (как делает LAPACK)
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i)
        perm[i] = i;

    for (int i = 0; i < n; ++i)
        std::swap(perm[i], perm[ipiv[i]]);

    for (int i = 0; i < n; ++i)
        P(i, perm[i]) = 1.0;

    return {P, L, U};
}

// 2. Блочный алгоритм LU-разложения
std::tuple<Matrix, Matrix, Matrix>
Matrix::lu_blocked(int mb, int nb) const
{
    if (rows_ != cols_)
        throw std::runtime_error("LU возможно только для квадратных матриц.");

    const int n = rows_;
    if (nb <= 1)
        nb = 64;
    if (mb <= 1)
        mb = 128;

    Matrix A(*this);
    double *a = A.data();
    const int ld = cols_;

    std::vector<int> ipiv(n);
    for (int i = 0; i < n; ++i)
        ipiv[i] = i;

    auto Aat = [&](int i, int j) -> double &
    {
        return a[(size_t)i * ld + j];
    };

    // ===== Основной блочный цикл =====
    for (int j = 0; j < n; j += nb)
    {
        int jb = std::min(nb, n - j);

        // ===== 1. Факторизация панели =====
        for (int col = j; col < j + jb; ++col)
        {
            // поиск pivot
            int p = col;
            double maxv = std::abs(Aat(col, col));
            for (int i = col + 1; i < n; ++i)
            {
                double v = std::abs(Aat(i, col));
                if (v > maxv)
                {
                    maxv = v;
                    p = i;
                }
            }

            if (maxv < 1e-15)
                throw std::runtime_error("Матрица вырождена.");

            ipiv[col] = p;

            // swap строк по ВСЕЙ матрице
            if (p != col)
            {
                for (int k = 0; k < n; ++k)
                    std::swap(Aat(col, k), Aat(p, k));
            }

            // L множители
            double inv = 1.0 / Aat(col, col);
            for (int i = col + 1; i < n; ++i)
                Aat(i, col) *= inv;

            // обновление панели
            for (int i = col + 1; i < n; ++i)
            {
                double lij = Aat(i, col);
                for (int k = col + 1; k < j + jb; ++k)
                    Aat(i, k) -= lij * Aat(col, k);
            }
        }

        if (j + jb >= n)
            continue;

        // ===== 2. TRSM (блок строки U) =====
        for (int i = j; i < j + jb; ++i)
        {
            for (int k = j; k < i; ++k)
            {
                double lik = Aat(i, k);
                for (int col = j + jb; col < n; ++col)
                    Aat(i, col) -= lik * Aat(k, col);
            }
        }

        // ===== 3. GEMM обновление хвоста =====
        for (int ii = j + jb; ii < n; ii += mb)
        {
            int i_end = std::min(ii + mb, n);

            for (int jj = j + jb; jj < n; jj += nb)
            {
                int j_end = std::min(jj + nb, n);

                for (int i = ii; i < i_end; ++i)
                {
                    for (int k = j; k < j + jb; ++k)
                    {
                        double lik = Aat(i, k);
                        for (int col = jj; col < j_end; ++col)
                            Aat(i, col) -= lik * Aat(k, col);
                    }
                }
            }
        }
    }

    // ===== Извлечение L и U =====
    Matrix L(n, n);
    Matrix U(n, n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i > j)
            {
                L(i, j) = A(i, j);
                U(i, j) = 0.0;
            }
            else if (i == j)
            {
                L(i, j) = 1.0;
                U(i, j) = A(i, j);
            }
            else
            {
                L(i, j) = 0.0;
                U(i, j) = A(i, j);
            }
        }
    }

    // ===== Построение P =====
    Matrix P(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            P(i, j) = 0.0;

    // perm как в LAPACK
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i)
        perm[i] = i;

    for (int i = 0; i < n; ++i)
        std::swap(perm[i], perm[ipiv[i]]);

    for (int i = 0; i < n; ++i)
        P(i, perm[i]) = 1.0;

    return {P, L, U};
}

// 3. Блочный параллельный алгоритм LU-разложения с OpenMP
std::tuple<Matrix, Matrix, Matrix> Matrix::lu_blocked_parallel(int block_size) const
{
    // Заглушка: просто вызываем последовательную версию
    // TODO: Implement parallel version
    return lu_simple();
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

    const double *A_data = data_; // исходная матрица
    double *L_data = L.data();    // матрица результата

    for (int i = 0; i < n; i++)
    {
        double *L_i = L_data + i * n; // строка i матрицы L

        for (int j = 0; j <= i; j++)
        {
            double *L_j = L_data + j * n; // строка j матрицы L
            double sum = 0.0;

// Векторизованное вычисление суммы
#pragma omp simd reduction(+ : sum)
            for (int k = 0; k < j; k++)
            {
                sum += L_i[k] * L_j[k];
            }

            if (i == j)
            {
                double diagonal_val = A_data[i * n + i] - sum;
                // if (diagonal_val <= 0)
                // {
                //     throw std::runtime_error("Матрица не является положительно-определенной.");
                // }
                L_i[i] = std::sqrt(diagonal_val);
            }
            else
            {
                // if (L_j[j] == 0)
                // {
                //     throw std::runtime_error("Деление на ноль в разложении Холецкого.");
                // }
                L_i[j] = (A_data[i * n + j] - sum) / L_j[j];
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

    int m = rows_;
    int n = other.cols_;
    int k = cols_;

    Matrix result(m, n);

    const double *A = data_;
    const double *B = other.data_;
    double *C = result.data();

    const int BLOCK = 128;

#pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < m; i0 += BLOCK)
    {
        for (int k0 = 0; k0 < k; k0 += BLOCK)
        {
            for (int j0 = 0; j0 < n; j0 += BLOCK)
            {
                int i_end = std::min(i0 + BLOCK, m);
                int k_end = std::min(k0 + BLOCK, k);
                int j_end = std::min(j0 + BLOCK, n);

                // Умножение блоков A[i0:i_end, k0:k_end] * B[k0:k_end, j0:j_end]
                for (int i = i0; i < i_end; ++i)
                {
                    double *C_i = C + i * n;
                    const double *A_i = A + i * k;

                    for (int kk = k0; kk < k_end; ++kk)
                    {
                        double aik = A_i[kk];
                        const double *B_k = B + kk * n;

                        for (int j = j0; j < j_end; ++j)
                        {
                            C_i[j] += aik * B_k[j];
                        }
                    }
                }
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

// Блочный алгоритм разложения Холецкого
Matrix Matrix::cholesky_blocked(int bs, int mb, int nb) const
{
    if (rows_ != cols_)
        throw std::runtime_error("Cholesky only for square matrices.");

    const int n = rows_;
    if (bs <= 0)
        bs = 64;
    if (mb <= 0)
        mb = 128;
    if (nb <= 0)
        nb = 64;

    Matrix R = *this;
    double *__restrict A = R.data();
    const int ld = n;

    auto row = [&](int i) -> double *
    {
        return A + (size_t)i * ld;
    };

    for (int k = 0; k < n; k += bs)
    {
        const int kend = std::min(k + bs, n);

        // 1) Factorize diagonal block (unblocked inside block)
        for (int j = k; j < kend; ++j)
        {
            double *Aj = row(j);

            double d = Aj[j];
            for (int p = k; p < j; ++p)
            {
                double v = Aj[p];
                d -= v * v;
            }
            // if (d <= 0.0) throw std::runtime_error("Matrix is not SPD.");

            d = std::sqrt(d);
            Aj[j] = d;
            const double inv = 1.0 / d;

            for (int i = j + 1; i < kend; ++i)
            {
                double *Ai = row(i);
                double s = Ai[j];
                for (int p = k; p < j; ++p)
                {
                    s -= Ai[p] * Aj[p];
                }
                Ai[j] = s * inv;
            }
        }

        if (kend >= n)
            break;

        // 2) Compute L21 (TRSM-like): rows kend..n-1, cols k..kend-1
        for (int j = k; j < kend; ++j)
        {
            double *Aj = row(j);
            const double ljj = Aj[j];

            for (int i = kend; i < n; ++i)
            {
                double *Ai = row(i);
                double s = Ai[j];
                for (int p = k; p < j; ++p)
                    s -= Ai[p] * Aj[p];
                Ai[j] = s / ljj;
            }
        }

        // 3) Update trailing A22 -= L21 * L21^T   (tiled mb×nb)
        for (int ii = kend; ii < n; ii += mb)
        {
            const int i_end = std::min(ii + mb, n);

            for (int jj = kend; jj <= ii; jj += nb) // only lower triangle tiles
            {
                const int j_end = std::min(jj + nb, n);

                for (int i = ii; i < i_end; ++i)
                {
                    double *Ai = row(i);
                    const int j_lim = std::min(j_end, i + 1);

                    for (int p = k; p < kend; ++p)
                    {
                        const double lip = Ai[p];

// inner loop over j contiguous in memory (row-major)
#pragma omp simd
                        for (int j = jj; j < j_lim; ++j)
                        {
                            Ai[j] -= lip * row(j)[p]; // A(j,p) is L21(j,p)
                        }
                    }
                }
            }
        }
    }

    // zero upper triangle
    for (int i = 0; i < n; ++i)
    {
        double *Ai = row(i);
        memset(Ai + i + 1, 0, (n - i - 1) * sizeof(double));
    }

    return R;
}

// Блочный параллельный алгоритм разложения Холецкого
Matrix Matrix::cholesky_blocked_parallel(int bs, int mb, int nb) const
{
    // Заглушка: просто вызываем последовательную блочную версию
    // TODO: Implement parallel version
    // return cholesky_blocked(block_size, block_size, block_size);
    if (rows_ != cols_)
        throw std::runtime_error("Cholesky only for square matrices.");

    const int n = rows_;
    if (bs <= 0)
        bs = 64;
    if (mb <= 0)
        mb = 128;
    if (nb <= 0)
        nb = 64;

    Matrix R = *this;
    double *__restrict A = R.data();
    const int ld = n;

    auto row = [&](int i) -> double *
    {
        return A + (size_t)i * ld;
    };

    for (int k = 0; k < n; k += bs)
    {
        const int kend = std::min(k + bs, n);

        // 1) Factorize diagonal block (unblocked inside block)
        for (int j = k; j < kend; ++j)
        {
            double *Aj = row(j);

            double d = Aj[j];
            for (int p = k; p < j; ++p)
            {
                double v = Aj[p];
                d -= v * v;
            }
            // if (d <= 0.0) throw std::runtime_error("Matrix is not SPD.");

            d = std::sqrt(d);
            Aj[j] = d;
            const double inv = 1.0 / d;

            for (int i = j + 1; i < kend; ++i)
            {
                double *Ai = row(i);
                double s = Ai[j];
                for (int p = k; p < j; ++p)
                {
                    s -= Ai[p] * Aj[p];
                }
                Ai[j] = s * inv;
            }
        }

        if (kend >= n)
            break;

        // 2) Compute L21 (TRSM-like): rows kend..n-1, cols k..kend-1
        for (int j = k; j < kend; ++j)
        {
            double *Aj = row(j);
            const double ljj = Aj[j];

            for (int i = kend; i < n; ++i)
            {
                double *Ai = row(i);
                double s = Ai[j];
                for (int p = k; p < j; ++p)
                    s -= Ai[p] * Aj[p];
                Ai[j] = s / ljj;
            }
        }

        // 3) Update trailing A22 -= L21 * L21^T   (tiled mb×nb)
        for (int ii = kend; ii < n; ii += mb)
        {
            const int i_end = std::min(ii + mb, n);

            for (int jj = kend; jj <= ii; jj += nb) // only lower triangle tiles
            {
                const int j_end = std::min(jj + nb, n);

                for (int i = ii; i < i_end; ++i)
                {
                    double *Ai = row(i);
                    const int j_lim = std::min(j_end, i + 1);

                    for (int p = k; p < kend; ++p)
                    {
                        const double lip = Ai[p];

// inner loop over j contiguous in memory (row-major)
#pragma omp simd
                        for (int j = jj; j < j_lim; ++j)
                        {
                            Ai[j] -= lip * row(j)[p]; // A(j,p) is L21(j,p)
                        }
                    }
                }
            }
        }
    }

    // zero upper triangle
    for (int i = 0; i < n; ++i)
    {
        double *Ai = row(i);
        memset(Ai + i + 1, 0, (n - i - 1) * sizeof(double));
    }

    return R;
}