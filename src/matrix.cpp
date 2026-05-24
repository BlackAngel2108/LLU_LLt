#include "matrix.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <numeric>   // Для std::iota
#include <cstring>   // Для std::memcpy
#include <algorithm> // для std::fill, std::copy
//#include <immintrin.h>
#include <tuple>
#include <omp.h>
#include <chrono>


#ifdef USE_RVV
#include <riscv_vector.h>
#endif
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

void Matrix::lu_simple(Matrix &P, Matrix &L, Matrix &U) const
{
    if (rows_ != cols_)
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");

    const int n = rows_;

    // Проверяем размеры выходных матриц
    if (P.getRows() != n || P.getCols() != n ||
        L.getRows() != n || L.getCols() != n ||
        U.getRows() != n || U.getCols() != n)
    {
        throw std::runtime_error("Output matrices P, L, U must have the same dimensions as the input matrix.");
    }

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
}

// 2. Блочный алгоритм LU-разложения
void Matrix::lu_blocked(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const
{
    if (rows_ != cols_)
        throw std::runtime_error("LU возможно только для квадратных матриц.");

    const int n = rows_;

    // Проверяем размеры выходных матриц
    if (P.getRows() != n || P.getCols() != n ||
        L.getRows() != n || L.getCols() != n ||
        U.getRows() != n || U.getCols() != n)
    {
        throw std::runtime_error("Output matrices P, L, U must have the same dimensions as the input matrix.");
    }

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
}

// 3. Блочный параллельный алгоритм LU-разложения с OpenMP
void Matrix::lu_blocked_parallel(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const
{
    if (rows_ != cols_)
        throw std::runtime_error("LU возможно только для квадратных матриц.");

    const int n = rows_;

    if (P.getRows() != n || P.getCols() != n ||
        L.getRows() != n || L.getCols() != n ||
        U.getRows() != n || U.getCols() != n)
    {
        throw std::runtime_error("Output matrices P, L, U must have the same dimensions.");
    }

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

    for (int j = 0; j < n; j += nb)
    {
        int jb = std::min(nb, n - j);

        // ===== 1. Факторизация панели (с векторизацией) =====
        for (int col = j; col < j + jb; ++col)
        {
            // поиск pivot (без векторизации)
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

            if (p != col)
            {
                for (int k = 0; k < n; ++k)
                    std::swap(Aat(col, k), Aat(p, k));
            }

            double inv = 1.0 / Aat(col, col);
            for (int i = col + 1; i < n; ++i)
                Aat(i, col) *= inv;

            // ВЕКТОРИЗОВАННОЕ обновление панели
            double *row_col = a + (size_t)col * ld;
            for (int i = col + 1; i < n; ++i)
            {
                double lij = Aat(i, col);
                double *row_i = a + (size_t)i * ld;
#pragma omp simd
                for (int k = col + 1; k < j + jb; ++k)
                    row_i[k] -= lij * row_col[k];
            }
        }

        if (j + jb >= n)
            continue;

// ===== 2. TRSM с векторизацией =====
#pragma omp parallel for schedule(static)
        for (int i = j; i < j + jb; ++i)
        {
            double *row_i = a + (size_t)i * ld;
            for (int k = j; k < i; ++k)
            {
                double lik = row_i[k];
                double *row_k = a + (size_t)k * ld;
#pragma omp simd
                for (int col = j + jb; col < n; ++col)
                    row_i[col] -= lik * row_k[col];
            }
        }

// ===== 3. GEMM с векторизацией =====
#pragma omp parallel for collapse(2) schedule(static)
        for (int ii = j + jb; ii < n; ii += mb)
        {
            for (int jj = j + jb; jj < n; jj += nb)
            {
                int i_end = std::min(ii + mb, n);
                int j_end = std::min(jj + nb, n);

                for (int i = ii; i < i_end; ++i)
                {
                    double *row_i = a + (size_t)i * ld;
                    for (int k = j; k < j + jb; ++k)
                    {
                        double lik = row_i[k];
                        double *row_k = a + (size_t)k * ld;
#pragma omp simd
                        for (int col = jj; col < j_end; ++col)
                            row_i[col] -= lik * row_k[col];
                    }
                }
            }
        }
    }

// ===== Извлечение L и U с векторизацией =====
#pragma omp parallel for
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
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            P(i, j) = 0.0;

    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i)
        perm[i] = i;
    for (int i = 0; i < n; ++i)
        std::swap(perm[i], perm[ipiv[i]]);

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        P(i, perm[i]) = 1.0;
}

// 4. Разложение Холецкого (LLt)
void Matrix::cholesky(Matrix &L) const
{
    if (rows_ != cols_)
    {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }

    int n = rows_;

    // Проверяем, что матрица L имеет правильный размер
    if (L.getRows() != n || L.getCols() != n)
    {
        throw std::runtime_error("Output matrix L must have the same dimensions.");
    }

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
}

// Умножение матриц
Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols_ != other.rows_)
    {
        throw std::invalid_argument("Несоответствие размеров матриц для умножения.");
    }

    const int m = rows_;
    const int n = other.cols_;
    const int k = cols_;
    const int BLOCK = 128; // Можно подобрать под кэш L2: ~256KB / 8 = ~128

    Matrix result(m, n);
    // Не обнуляем result здесь - обнулим блоками при накоплении

    const double *A = data_;
    const double *B = other.data_;
    double *C = result.data();

    // Оптимизация 1: обнуляем результат через memset (быстро)
    memset(C, 0, result.size() * sizeof(double));

// Оптимизация 2: правильный порядок блоков i0, j0, k0
#pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < m; i0 += BLOCK)
    {
        for (int j0 = 0; j0 < n; j0 += BLOCK)
        {
            const int i_end = std::min(i0 + BLOCK, m);
            const int j_end = std::min(j0 + BLOCK, n);

            // Оптимизация 3: накопление через блоки k
            for (int k0 = 0; k0 < k; k0 += BLOCK)
            {
                const int k_end = std::min(k0 + BLOCK, k);

                // Оптимизация 4: ручная размотка для улучшения ILP
                for (int i = i0; i < i_end; ++i)
                {
                    double *C_i = C + i * n;
                    const double *A_i = A + i * k;

                    // Оптимизация 5: размотка цикла по kk (по 4 элемента)
                    int kk = k0;
                    for (; kk + 3 < k_end; kk += 4)
                    {
                        double aik0 = A_i[kk];
                        double aik1 = A_i[kk + 1];
                        double aik2 = A_i[kk + 2];
                        double aik3 = A_i[kk + 3];

                        const double *B_k0 = B + kk * n;
                        const double *B_k1 = B + (kk + 1) * n;
                        const double *B_k2 = B + (kk + 2) * n;
                        const double *B_k3 = B + (kk + 3) * n;

#pragma omp simd
                        for (int j = j0; j < j_end; ++j)
                        {
                            C_i[j] += aik0 * B_k0[j] + aik1 * B_k1[j] +
                                      aik2 * B_k2[j] + aik3 * B_k3[j];
                        }
                    }

                    // Остаток
                    for (; kk < k_end; ++kk)
                    {
                        double aik = A_i[kk];
                        const double *B_k = B + kk * n;

#pragma omp simd
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
void Matrix::cholesky_blocked(Matrix &L, int bs) const
{
    if (rows_ != cols_)
        throw std::runtime_error("Cholesky only for square matrices.");

    const int n = rows_;

    if (bs <= 0)
        bs = 48;

    if (L.getRows() != n || L.getCols() != n)
    {
        throw std::runtime_error("Output matrix L must have the same dimensions.");
    }

    // Копируем текущую матрицу в L
    L = *this;
    const int ld = L.getCols();

    // Обнуляем верхний треугольник
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            L.at_unchecked(i, j) = 0.0;
        }
    }

    const int blocks = (n + bs - 1) / bs;

    for (int bk = 0; bk < blocks; ++bk)
    {
        int k = bk * bs;
        int bs_current = std::min(bs, n - k);
        int kend = k + bs_current;

        // Фаза 1: Факторизация диагонального блока
        for (int i = k; i < kend; ++i)
        {
            for (int j = k; j < i; ++j)
            {
                double s = L.at_unchecked(i, j);
                for (int p = k; p < j; ++p)
                {
                    s -= L.at_unchecked(i, p) * L.at_unchecked(j, p);
                }
                L.at_unchecked(i, j) = s / L.at_unchecked(j, j);
            }

            double diag = L.at_unchecked(i, i);
            for (int p = k; p < i; ++p)
            {
                double v = L.at_unchecked(i, p);
                diag -= v * v;
            }

            if (diag < 0.0 && diag > -1e-12)
            {
                diag = 0.0;
            }

            if (diag <= 0.0)
            {
                throw std::runtime_error("Matrix is not SPD (diagonal <= 0)");
            }

            L.at_unchecked(i, i) = std::sqrt(diag);
        }

        if (kend >= n)
            break;

        // Фаза 2: Вычисление блока L21
        for (int bi = bk + 1; bi < blocks; ++bi)
        {
            int i0 = bi * bs;
            int ib = std::min(bs, n - i0);
            int iend = i0 + ib;

            for (int i = i0; i < iend; ++i)
            {
                for (int j = k; j < kend; ++j)
                {
                    double s = L.at_unchecked(i, j);
                    for (int p = k; p < j; ++p)
                    {
                        s -= L.at_unchecked(i, p) * L.at_unchecked(j, p);
                    }
                    L.at_unchecked(i, j) = s / L.at_unchecked(j, j);
                }
            }
        }

        // Фаза 3a: Обновление диагональных блоков
        for (int bi = bk + 1; bi < blocks; ++bi)
        {
            int i0 = bi * bs;
            int ib = std::min(bs, n - i0);
            int iend = i0 + ib;

            for (int i = i0; i < iend; ++i)
            {
                for (int j = i0; j <= i; ++j)
                {
                    double s = L.at_unchecked(i, j);
                    for (int p = k; p < kend; ++p)
                    {
                        s -= L.at_unchecked(i, p) * L.at_unchecked(j, p);
                    }
                    L.at_unchecked(i, j) = s;
                }
            }
        }

        // Фаза 3b: Обновление off-diagonal блоков
        for (int bi = bk + 1; bi < blocks; ++bi)
        {
            for (int bj = bk + 1; bj < bi; ++bj)
            {
                int i0 = bi * bs;
                int ib = std::min(bs, n - i0);
                int iend = i0 + ib;

                int j0 = bj * bs;
                int jb = std::min(bs, n - j0);
                int jend = j0 + jb;

                for (int i = i0; i < iend; ++i)
                {
                    for (int j = j0; j < jend; ++j)
                    {
                        double s = L.at_unchecked(i, j);
                        for (int p = k; p < kend; ++p)
                        {
                            s -= L.at_unchecked(i, p) * L.at_unchecked(j, p);
                        }
                        L.at_unchecked(i, j) = s;
                    }
                }
            }
        }
    }
}

// Блочный параллельный алгоритм разложения Холецкого
void Matrix::cholesky_blocked_parallel(Matrix &L, int bs) const
{
    if (rows_ != cols_)
        throw std::runtime_error("Cholesky only for square matrices.");

    const int n = rows_;

    if (bs <= 0)
        bs = 48; // Оптимальный размер блока из эталонного кода

    if (L.getRows() != n || L.getCols() != n)
    {
        throw std::runtime_error("Output matrix L must have the same dimensions.");
    }

    double *A = L.data();
    const int ld = L.getCols();

    // Вспомогательная функция индексации (row-major)
    auto idx = [ld](int i, int j) -> size_t
    {
        return (size_t)i * ld + j;
    };

// ===== Инициализация L: копируем нижний треугольник, обнуляем верхний =====
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            A[idx(i, j)] = (*this)(i, j);
        }
        for (int j = i + 1; j < n; ++j)
        {
            A[idx(i, j)] = 0.0;
        }
    }

    const int blocks = (n + bs - 1) / bs;

    // ===== Основной блочный цикл =====
    for (int bk = 0; bk < blocks; ++bk)
    {
        int k = bk * bs;
        int bs_current = std::min(bs, n - k);
        int kend = k + bs_current;

        // ===== Фаза 1: Факторизация диагонального блока (последовательно) =====
        for (int i = k; i < kend; ++i)
        {
            for (int j = k; j < i; ++j)
            {
                double s = A[idx(i, j)];
                for (int p = k; p < j; ++p)
                {
                    s -= A[idx(i, p)] * A[idx(j, p)];
                }
                A[idx(i, j)] = s / A[idx(j, j)];
            }

            double diag = A[idx(i, i)];
            for (int p = k; p < i; ++p)
            {
                double v = A[idx(i, p)];
                diag -= v * v;
            }

            if (diag < 0.0 && diag > -1e-12)
            {
                diag = 0.0;
            }

            if (diag <= 0.0)
            {
                throw std::runtime_error("Matrix is not SPD");
            }

            A[idx(i, i)] = std::sqrt(diag);
        }

        if (kend >= n)
            break;

// ===== Фаза 2: Вычисление блока L21 (параллельно) =====
#pragma omp parallel for schedule(static)
        for (int bi = bk + 1; bi < blocks; ++bi)
        {
            int i0 = bi * bs;
            int iend = std::min(i0 + bs, n);

            for (int i = i0; i < iend; ++i)
            {
                for (int j = k; j < kend; ++j)
                {
                    double s = A[idx(i, j)];
                    for (int p = k; p < j; ++p)
                    {
                        s -= A[idx(i, p)] * A[idx(j, p)];
                    }
                    A[idx(i, j)] = s / A[idx(j, j)];
                }
            }
        }

// ===== Фаза 3: Обновление trailing matrix (параллельно) =====
#pragma omp parallel for schedule(static)
        for (int bi = bk + 1; bi < blocks; ++bi)
        {
            int i0 = bi * bs;
            int iend = std::min(i0 + bs, n);

            // Диагональные блоки
            for (int i = i0; i < iend; ++i)
            {
                for (int j = i0; j <= i; ++j)
                {
                    double s = A[idx(i, j)];
                    for (int p = k; p < kend; ++p)
                    {
                        s -= A[idx(i, p)] * A[idx(j, p)];
                    }
                    A[idx(i, j)] = s;
                }
            }

            // Off-diagonal блоки
            for (int bj = bk + 1; bj < bi; ++bj)
            {
                int j0 = bj * bs;
                int jend = std::min(j0 + bs, n);

                for (int i = i0; i < iend; ++i)
                {
                    for (int j = j0; j < jend; ++j)
                    {
                        double s = A[idx(i, j)];
                        for (int p = k; p < kend; ++p)
                        {
                            s -= A[idx(i, p)] * A[idx(j, p)];
                        }
                        A[idx(i, j)] = s;
                    }
                }
            }
        }
    }
}


#ifdef USE_RVV

// ============================================================
// RVV helper kernels: y[col] -= alpha * x[col]
// ============================================================

static inline void rvv_axpy_neg_m1(double *y, const double *x, double alpha, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m1(len);

        vfloat64m1_t vy = __riscv_vle64_v_f64m1(y + off, vl);
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(x + off, vl);

        vy = __riscv_vfmacc_vf_f64m1(vy, -alpha, vx, vl);

        __riscv_vse64_v_f64m1(y + off, vy, vl);

        off += vl;
        len -= vl;
    }
}

static inline void rvv_axpy_neg_m2(double *y, const double *x, double alpha, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m2(len);

        vfloat64m2_t vy = __riscv_vle64_v_f64m2(y + off, vl);
        vfloat64m2_t vx = __riscv_vle64_v_f64m2(x + off, vl);

        vy = __riscv_vfmacc_vf_f64m2(vy, -alpha, vx, vl);

        __riscv_vse64_v_f64m2(y + off, vy, vl);

        off += vl;
        len -= vl;
    }
}

static inline void rvv_axpy_neg_m4(double *y, const double *x, double alpha, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m4(len);

        vfloat64m4_t vy = __riscv_vle64_v_f64m4(y + off, vl);
        vfloat64m4_t vx = __riscv_vle64_v_f64m4(x + off, vl);

        vy = __riscv_vfmacc_vf_f64m4(vy, -alpha, vx, vl);

        __riscv_vse64_v_f64m4(y + off, vy, vl);

        off += vl;
        len -= vl;
    }
}

static inline void rvv_axpy_neg_m8(double *y, const double *x, double alpha, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m8(len);

        vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + off, vl);
        vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + off, vl);

        vy = __riscv_vfmacc_vf_f64m8(vy, -alpha, vx, vl);

        __riscv_vse64_v_f64m8(y + off, vy, vl);

        off += vl;
        len -= vl;
    }
}

// ============================================================
// RVV helper kernels: dot product
// ============================================================

static inline double rvv_dot_m1(const double *a, const double *b, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0.0, vlmax);

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m1(len);

        vfloat64m1_t va = __riscv_vle64_v_f64m1(a + off, vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(b + off, vl);

        vsum = __riscv_vfmacc_vv_f64m1(vsum, va, vb, vl);

        off += vl;
        len -= vl;
    }

    vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t vred = __riscv_vfredusum_vs_f64m1_f64m1(vsum, vzero, vlmax);

    double result = 0.0;
    __riscv_vse64_v_f64m1(&result, vred, 1);

    return result;
}

static inline double rvv_dot_m2(const double *a, const double *b, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t vsum = __riscv_vfmv_v_f_f64m2(0.0, vlmax);

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m2(len);

        vfloat64m2_t va = __riscv_vle64_v_f64m2(a + off, vl);
        vfloat64m2_t vb = __riscv_vle64_v_f64m2(b + off, vl);

        vsum = __riscv_vfmacc_vv_f64m2(vsum, va, vb, vl);

        off += vl;
        len -= vl;
    }

    vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t vred = __riscv_vfredusum_vs_f64m2_f64m1(vsum, vzero, vlmax);

    double result = 0.0;
    __riscv_vse64_v_f64m1(&result, vred, 1);

    return result;
}

static inline double rvv_dot_m4(const double *a, const double *b, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t vsum = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m4(len);

        vfloat64m4_t va = __riscv_vle64_v_f64m4(a + off, vl);
        vfloat64m4_t vb = __riscv_vle64_v_f64m4(b + off, vl);

        vsum = __riscv_vfmacc_vv_f64m4(vsum, va, vb, vl);

        off += vl;
        len -= vl;
    }

    vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t vred = __riscv_vfredusum_vs_f64m4_f64m1(vsum, vzero, vlmax);

    double result = 0.0;
    __riscv_vse64_v_f64m1(&result, vred, 1);

    return result;
}

static inline double rvv_dot_m8(const double *a, const double *b, int begin, int end)
{
    int len = end - begin;
    int off = begin;

    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e64m8(len);

        vfloat64m8_t va = __riscv_vle64_v_f64m8(a + off, vl);
        vfloat64m8_t vb = __riscv_vle64_v_f64m8(b + off, vl);

        vsum = __riscv_vfmacc_vv_f64m8(vsum, va, vb, vl);

        off += vl;
        len -= vl;
    }

    vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, vzero, vlmax);

    double result = 0.0;
    __riscv_vse64_v_f64m1(&result, vred, 1);

    return result;
}

// ============================================================
// Common LU RVV implementation
// ============================================================

using RvvAxpyNegKernel = void (*)(double *, const double *, double, int, int);

static void lu_blocked_parallel_rvv_impl(
    const Matrix &src,
    Matrix &P,
    Matrix &L,
    Matrix &U,
    int mb,
    int nb,
    RvvAxpyNegKernel rvv_axpy_neg)
{
    if (src.getRows() != src.getCols()) {
        throw std::runtime_error("LU возможно только для квадратных матриц.");
    }

    const int n = src.getRows();

    if (P.getRows() != n || P.getCols() != n ||
        L.getRows() != n || L.getCols() != n ||
        U.getRows() != n || U.getCols() != n) {
        throw std::runtime_error("Output matrices P, L, U must have the same dimensions.");
    }

    if (nb <= 1) nb = 64;
    if (mb <= 1) mb = 128;

    Matrix A(src);
    double *a = A.data();
    const int ld = A.getCols();

    auto idx = [ld](int i, int j) -> size_t {
        return static_cast<size_t>(i) * ld + j;
    };

    auto Aat = [&](int i, int j) -> double & {
        return a[idx(i, j)];
    };

    std::vector<int> ipiv(n);
    for (int i = 0; i < n; ++i) {
        ipiv[i] = i;
    }

    for (int j = 0; j < n; j += nb) {
        int jb = std::min(nb, n - j);

        // 1. Факторизация панели
        for (int col = j; col < j + jb; ++col) {
            int p = col;
            double maxv = std::abs(Aat(col, col));

            for (int i = col + 1; i < n; ++i) {
                double v = std::abs(Aat(i, col));
                if (v > maxv) {
                    maxv = v;
                    p = i;
                }
            }

            if (maxv < 1e-15) {
                throw std::runtime_error("Матрица вырождена.");
            }

            ipiv[col] = p;

            if (p != col) {
                for (int k = 0; k < n; ++k) {
                    std::swap(Aat(col, k), Aat(p, k));
                }
            }

            double inv = 1.0 / Aat(col, col);

            for (int i = col + 1; i < n; ++i) {
                Aat(i, col) *= inv;
            }

            double *row_col = a + idx(col, 0);

            for (int i = col + 1; i < n; ++i) {
                double lij = Aat(i, col);
                double *row_i = a + idx(i, 0);

                rvv_axpy_neg(row_i, row_col, lij, col + 1, j + jb);
            }
        }

        if (j + jb >= n) {
            continue;
        }

        // 2. TRSM / обновление блока строки U
        #pragma omp parallel for schedule(static)
        for (int i = j; i < j + jb; ++i) {
            double *row_i = a + idx(i, 0);

            for (int k = j; k < i; ++k) {
                double lik = row_i[k];
                const double *row_k = a + idx(k, 0);

                rvv_axpy_neg(row_i, row_k, lik, j + jb, n);
            }
        }

        // 3. GEMM обновление trailing matrix
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ii = j + jb; ii < n; ii += mb) {
            for (int jj = j + jb; jj < n; jj += nb) {
                int i_end = std::min(ii + mb, n);
                int j_end = std::min(jj + nb, n);

                for (int i = ii; i < i_end; ++i) {
                    double *row_i = a + idx(i, 0);

                    for (int k = j; k < j + jb; ++k) {
                        double lik = row_i[k];
                        const double *row_k = a + idx(k, 0);

                        rvv_axpy_neg(row_i, row_k, lik, jj, j_end);
                    }
                }
            }
        }
    }

    // Извлечение L и U
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j2 = 0; j2 < n; ++j2) {
            if (i > j2) {
                L(i, j2) = A(i, j2);
                U(i, j2) = 0.0;
            } else if (i == j2) {
                L(i, j2) = 1.0;
                U(i, j2) = A(i, j2);
            } else {
                L(i, j2) = 0.0;
                U(i, j2) = A(i, j2);
            }
        }
    }

    // Построение P
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j2 = 0; j2 < n; ++j2) {
            P(i, j2) = 0.0;
        }
    }

    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) {
        perm[i] = i;
    }

    for (int i = 0; i < n; ++i) {
        std::swap(perm[i], perm[ipiv[i]]);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        P(i, perm[i]) = 1.0;
    }
}

void Matrix::lu_blocked_parallel_rvv_m1(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const
{
    lu_blocked_parallel_rvv_impl(*this, P, L, U, mb, nb, rvv_axpy_neg_m1);
}

void Matrix::lu_blocked_parallel_rvv_m2(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const
{
    lu_blocked_parallel_rvv_impl(*this, P, L, U, mb, nb, rvv_axpy_neg_m2);
}

void Matrix::lu_blocked_parallel_rvv_m4(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const
{
    lu_blocked_parallel_rvv_impl(*this, P, L, U, mb, nb, rvv_axpy_neg_m4);
}

void Matrix::lu_blocked_parallel_rvv_m8(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const
{
    lu_blocked_parallel_rvv_impl(*this, P, L, U, mb, nb, rvv_axpy_neg_m8);
}

// ============================================================
// Common Cholesky / LLt RVV implementation
// ============================================================

using RvvDotKernel = double (*)(const double *, const double *, int, int);

static void cholesky_blocked_parallel_rvv_impl(
    const Matrix &src,
    Matrix &L,
    int bs,
    RvvDotKernel rvv_dot)
{
    if (src.getRows() != src.getCols()) {
        throw std::runtime_error("Cholesky only for square matrices.");
    }

    const int n = src.getRows();

    if (bs <= 0) {
        bs = 48;
    }

    if (L.getRows() != n || L.getCols() != n) {
        throw std::runtime_error("Output matrix L must have the same dimensions.");
    }

    double *A = L.data();
    const int ld = L.getCols();

    auto idx = [ld](int i, int j) -> size_t {
        return static_cast<size_t>(i) * ld + j;
    };

    // Копируем нижний треугольник, верхний зануляем
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            A[idx(i, j)] = src(i, j);
        }

        for (int j = i + 1; j < n; ++j) {
            A[idx(i, j)] = 0.0;
        }
    }

    const int blocks = (n + bs - 1) / bs;

    for (int bk = 0; bk < blocks; ++bk) {
        int k = bk * bs;
        int kend = std::min(k + bs, n);

        // 1. Факторизация диагонального блока
        for (int i = k; i < kend; ++i) {
            double *row_i = A + idx(i, 0);

            for (int j = k; j < i; ++j) {
                double *row_j = A + idx(j, 0);

                double s = row_i[j];
                s -= rvv_dot(row_i, row_j, k, j);

                row_i[j] = s / row_j[j];
            }

            double diag = row_i[i];
            diag -= rvv_dot(row_i, row_i, k, i);

            if (diag < 0.0 && diag > -1e-12) {
                diag = 0.0;
            }

            if (diag <= 0.0) {
                throw std::runtime_error("Matrix is not SPD");
            }

            row_i[i] = std::sqrt(diag);
        }

        if (kend >= n) {
            break;
        }

        // 2. Вычисление L21
        #pragma omp parallel for schedule(static)
        for (int bi = bk + 1; bi < blocks; ++bi) {
            int i0 = bi * bs;
            int iend = std::min(i0 + bs, n);

            for (int i = i0; i < iend; ++i) {
                double *row_i = A + idx(i, 0);

                for (int j = k; j < kend; ++j) {
                    double *row_j = A + idx(j, 0);

                    double s = row_i[j];
                    s -= rvv_dot(row_i, row_j, k, j);

                    row_i[j] = s / row_j[j];
                }
            }
        }

        // 3. Обновление trailing matrix
        #pragma omp parallel for schedule(static)
        for (int bi = bk + 1; bi < blocks; ++bi) {
            int i0 = bi * bs;
            int iend = std::min(i0 + bs, n);

            // Диагональный блок
            for (int i = i0; i < iend; ++i) {
                double *row_i = A + idx(i, 0);

                for (int j = i0; j <= i; ++j) {
                    double *row_j = A + idx(j, 0);

                    double s = row_i[j];
                    s -= rvv_dot(row_i, row_j, k, kend);

                    row_i[j] = s;
                }
            }

            // Off-diagonal блоки
            for (int bj = bk + 1; bj < bi; ++bj) {
                int j0 = bj * bs;
                int jend = std::min(j0 + bs, n);

                for (int i = i0; i < iend; ++i) {
                    double *row_i = A + idx(i, 0);

                    for (int j = j0; j < jend; ++j) {
                        double *row_j = A + idx(j, 0);

                        double s = row_i[j];
                        s -= rvv_dot(row_i, row_j, k, kend);

                        row_i[j] = s;
                    }
                }
            }
        }
    }
}

void Matrix::cholesky_blocked_parallel_rvv_m1(Matrix &L, int bs) const
{
    cholesky_blocked_parallel_rvv_impl(*this, L, bs, rvv_dot_m1);
}

void Matrix::cholesky_blocked_parallel_rvv_m2(Matrix &L, int bs) const
{
    cholesky_blocked_parallel_rvv_impl(*this, L, bs, rvv_dot_m2);
}

void Matrix::cholesky_blocked_parallel_rvv_m4(Matrix &L, int bs) const
{
    cholesky_blocked_parallel_rvv_impl(*this, L, bs, rvv_dot_m4);
}

void Matrix::cholesky_blocked_parallel_rvv_m8(Matrix &L, int bs) const
{
    cholesky_blocked_parallel_rvv_impl(*this, L, bs, rvv_dot_m8);
}

#else

void Matrix::lu_blocked_parallel_rvv_m1(Matrix &, Matrix &, Matrix &, int, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

void Matrix::lu_blocked_parallel_rvv_m2(Matrix &, Matrix &, Matrix &, int, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

void Matrix::lu_blocked_parallel_rvv_m4(Matrix &, Matrix &, Matrix &, int, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

void Matrix::lu_blocked_parallel_rvv_m8(Matrix &, Matrix &, Matrix &, int, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

void Matrix::cholesky_blocked_parallel_rvv_m1(Matrix &, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

void Matrix::cholesky_blocked_parallel_rvv_m2(Matrix &, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

void Matrix::cholesky_blocked_parallel_rvv_m4(Matrix &, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

void Matrix::cholesky_blocked_parallel_rvv_m8(Matrix &, int) const
{
    throw std::runtime_error("RVV support is disabled. Build with -DUSE_RVV and RISC-V vector flags.");
}

#endif