#include "matrix.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iomanip>

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Размеры матрицы должны быть положительными.");
    }
}

Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) {
        return *this;
    }
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
    return *this;
}

double& Matrix::operator()(int row, int col) {
    if (row >= rows_ || col >= cols_ || row < 0 || col < 0) {
        throw std::out_of_range("Индекс за пределами матрицы.");
    }
    return data_[row * cols_ + col];
}

const double& Matrix::operator()(int row, int col) const {
    if (row >= rows_ || col >= cols_ || row < 0 || col < 0) {
        throw std::out_of_range("Индекс за пределами матрицы.");
    }
    return data_[row * cols_ + col];
}

int Matrix::getRows() const {
    return rows_;
}

int Matrix::getCols() const {
    return cols_;
}

void Matrix::fillRandom() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 100.0);
    for (int i = 0; i < rows_ * cols_; ++i) {
        data_[i] = dis(gen);
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Реализация трех версий LU-разложения

// 1. Простой алгоритм LU-разложения
std::pair<Matrix, Matrix> Matrix::lu_simple() const {
    if (rows_ != cols_) {
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");
    }

    int n = rows_;
    Matrix L(n, n);
    Matrix U = *this;

    for (int i = 0; i < n; ++i) {
        L(i, i) = 1.0;
        for (int j = i + 1; j < n; ++j) {
            double factor = U(j, i) / U(i, i);
            L(j, i) = factor;
            for (int k = i; k < n; ++k) {
                U(j, k) -= factor * U(i, k);
            }
        }
    }
    return {L, U};
}

// 2. Блочный алгоритм LU-разложения
std::pair<Matrix, Matrix> Matrix::lu_blocked(int block_size) const {
    if (rows_ != cols_) {
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");
    }

    int n = rows_;
    Matrix A = *this;
    Matrix L(n, n);
    Matrix U(n, n);

    for (int k = 0; k < n; k += block_size) {
        int end_k = std::min(k + block_size, n);

        // Блок A11
        for (int i = k; i < end_k; ++i) {
            for (int j = i; j < end_k; ++j) {
                double sum = 0;
                for (int m = 0; m < i; ++m) {
                    sum += L(i, m) * U(m, j);
                }
                U(i, j) = A(i, j) - sum;
            }
            L(i, i) = 1.0;
            for (int j = i + 1; j < end_k; ++j) {
                double sum = 0;
                for (int m = 0; m < i; ++m) {
                    sum += L(j, m) * U(m, i);
                }
                L(j, i) = (A(j, i) - sum) / U(i, i);
            }
        }
        
        // Блоки A12 и A21
        for (int i = k; i < end_k; ++i) {
            for (int j = end_k; j < n; ++j) {
                double sum_u = 0;
                for(int m = 0; m < k; ++m) {
                    sum_u += L(i,m) * U(m,j);
                }
                U(i,j) = A(i,j) - sum_u;
            }
        }

        for (int i = end_k; i < n; ++i) {
            for (int j = k; j < end_k; ++j) {
                double sum_l = 0;
                for(int m = 0; m < k; ++m) {
                    sum_l += L(i,m) * U(m,j);
                }
                 if (U(j,j) == 0) continue;
                L(i,j) = (A(i,j) - sum_l) / U(j,j);
            }
        }

        // Обновление A22
        for (int i = end_k; i < n; ++i) {
            for (int j = end_k; j < n; ++j) {
                double sum_a = 0;
                for(int m = k; m < end_k; ++m) {
                    sum_a += L(i,m) * U(m,j);
                }
                A(i, j) -= sum_a;
            }
        }
    }
    return {L, U};
}


// 3. Блочный параллельный алгоритм LU-разложения с OpenMP
std::pair<Matrix, Matrix> Matrix::lu_blocked_parallel(int block_size) const {
    if (rows_ != cols_) {
        throw std::runtime_error("LU-разложение возможно только для квадратных матриц.");
    }
    
    int n = rows_;
    Matrix A = *this;
    Matrix L(n, n);
    Matrix U(n, n);

    for (int k = 0; k < n; k += block_size) {
        int end_k = std::min(k + block_size, n);
        
        // --- Этап 1: Факторизация диагонального блока (A11) ---
        // Этот этап выполняется последовательно, так как он критический
        for (int i = k; i < end_k; ++i) {
            for (int j = i; j < end_k; ++j) {
                double sum = 0;
                for (int m = 0; m < i; ++m) sum += L(i, m) * U(m, j);
                U(i, j) = A(i, j) - sum;
            }
            L(i, i) = 1.0;
            for (int j = i + 1; j < end_k; ++j) {
                double sum = 0;
                for (int m = 0; m < i; ++m) sum += L(j, m) * U(m, i);
                if (U(i, i) == 0) continue;
                L(j, i) = (A(j, i) - sum) / U(i, i);
            }
        }

        // --- Этап 2: Вычисление L21 и U12 ---
        // Эти два цикла можно распараллелить
        #pragma omp parallel for
        for (int i = end_k; i < n; ++i) { // L21
            for (int j = k; j < end_k; ++j) {
                double sum = 0;
                for (int m = 0; m < k; ++m) sum += L(i, m) * U(m, j);
                 if (U(j,j) == 0) continue;
                L(i, j) = (A(i, j) - sum) / U(j, j);
            }
        }
        
        #pragma omp parallel for
        for (int j = end_k; j < n; ++j) { // U12
             for (int i = k; i < end_k; ++i) {
                double sum = 0;
                for (int m = 0; m < k; ++m) sum += L(i, m) * U(m, j);
                U(i, j) = A(i, j) - sum;
            }
        }

        // --- Этап 3: Обновление оставшейся части матрицы (A22) ---
        // Этот цикл также хорошо параллелится
        #pragma omp parallel for
        for (int i = end_k; i < n; ++i) {
            for (int j = end_k; j < n; ++j) {
                double sum = 0;
                for (int m = k; m < end_k; ++m) {
                    sum += L(i, m) * U(m, j);
                }
                #pragma omp atomic
                A(i, j) -= sum;
            }
        }
    }

    return {L, U};
}

// 4. Разложение Холецкого (LLt)
Matrix Matrix::cholesky() const {
    if (rows_ != cols_) {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }
    int n = rows_;
    Matrix L(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            if (j == i) { // Диагональные элементы
                for (int k = 0; k < j; k++) {
                    sum += L(j, k) * L(j, k);
                }
                double diagonal_val = data_[j * n + j] - sum;
                if (diagonal_val <= 0) {
                    throw std::runtime_error("Матрица не является положительно-определенной.");
                }
                L(j, j) = std::sqrt(diagonal_val);
            } else { // Внедиагональные элементы
                for (int k = 0; k < j; k++) {
                    sum += L(i, k) * L(j, k);
                }
                if (L(j, j) == 0) {
                     throw std::runtime_error("Деление на ноль в разложении Холецкого.");
                }
                L(i, j) = (data_[i * n + j] - sum) / L(j, j);
            }
        }
    }

    return L;
}

// Умножение матриц
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Несоответствие размеров матриц для умножения.");
    }
    Matrix result(rows_, other.cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < other.cols_; ++j) {
            for (int k = 0; k < cols_; ++k) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return result;
}

// Транспонирование матрицы
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Блочный алгоритм разложения Холецкого
Matrix Matrix::cholesky_blocked(int block_size) const {
    if (rows_ != cols_) {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }
    int n = rows_;
    Matrix L(n, n);
    Matrix A = *this; // Копируем матрицу, чтобы изменять ее

    for (int k = 0; k < n; k += block_size) {
        int end = std::min(k + block_size, n);

        // 1. Факторизация диагонального блока A11
        for (int i = k; i < end; ++i) {
            for (int j = k; j <= i; ++j) {
                double sum = 0;
                for (int m = 0; m < j; ++m) {
                    sum += L(i, m) * L(j, m);
                }
                if (i == j) {
                    double diag_val = A(i, i) - sum;
                    if (diag_val <= 0) throw std::runtime_error("Матрица не положительно-определенная");
                    L(i, i) = std::sqrt(diag_val);
                } else {
                    if (L(j, j) == 0) throw std::runtime_error("Деление на ноль");
                    L(i, j) = (A(i, j) - sum) / L(j, j);
                }
            }
        }

        // 2. Вычисление поддиагонального блока L21
        for (int i = end; i < n; ++i) {
            for (int j = k; j < end; ++j) {
                double sum = 0;
                for (int m = 0; m < j; ++m) {
                    sum += L(i, m) * L(j, m);
                }
                if (L(j, j) == 0) throw std::runtime_error("Деление на ноль");
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }

        // 3. Обновление оставшейся части матрицы A22
        for (int i = end; i < n; ++i) {
            for (int j = end; j <= i; ++j) {
                double sum = 0;
                for (int m = k; m < end; ++m) {
                    sum += L(i, m) * L(j, m);
                }
                A(i, j) -= sum;
            }
        }
    }
    return L;
}

// Блочный параллельный алгоритм разложения Холецкого
Matrix Matrix::cholesky_blocked_parallel(int block_size) const {
    if (rows_ != cols_) {
        throw std::runtime_error("Разложение Холецкого возможно только для квадратных матриц.");
    }
    int n = rows_;
    Matrix L(n, n);
    Matrix A = *this;

    for (int k = 0; k < n; k += block_size) {
        int end = std::min(k + block_size, n);

        // 1. Факторизация диагонального блока A11 (последовательно)
        for (int i = k; i < end; ++i) {
            for (int j = k; j <= i; ++j) {
                double sum = 0;
                for (int m = 0; m < j; ++m) {
                    sum += L(i, m) * L(j, m);
                }
                if (i == j) {
                    double diag_val = A(i, i) - sum;
                    if (diag_val <= 0) throw std::runtime_error("Матрица не положительно-определенная");
                    L(i, i) = std::sqrt(diag_val);
                } else {
                    if (L(j, j) == 0) throw std::runtime_error("Деление на ноль");
                    L(i, j) = (A(i, j) - sum) / L(j, j);
                }
            }
        }

        // 2. Вычисление поддиагонального блока L21 (параллельно)
        #pragma omp parallel for
        for (int i = end; i < n; ++i) {
            for (int j = k; j < end; ++j) {
                double sum = 0;
                for (int m = 0; m < j; ++m) {
                    sum += L(i, m) * L(j, m);
                }
                if (L(j, j) == 0) {
                    // В параллельном коде бросать исключения небезопасно
                } else {
                    L(i, j) = (A(i, j) - sum) / L(j, j);
                }
            }
        }
        
        // 3. Обновление оставшейся части матрицы A22 (параллельно)
        #pragma omp parallel for
        for (int i = end; i < n; ++i) {
            for (int j = end; j <= i; ++j) {
                double sum = 0;
                for (int m = k; m < end; ++m) {
                    sum += L(i, m) * L(j, m);
                }
                #pragma omp atomic
                A(i, j) -= sum;
            }
        }
    }
    return L;
}
