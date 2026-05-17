#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cstddef> // для size_t

class Matrix
{
public:
    // Конструктор
    Matrix(int rows, int cols);
    // Конструктор копирования
    Matrix(const Matrix &other);
    // Конструктор перемещения (NEW)
    Matrix(Matrix &&other) noexcept;
    // Оператор присваивания
    Matrix &operator=(const Matrix &other);
    // Оператор перемещения (NEW)
    Matrix &operator=(Matrix &&other) noexcept;
    // Деструктор (NEW)
    ~Matrix();

    inline double *row_ptr(int r) { return data_ + (size_t)r * cols_; }
    inline const double *row_ptr(int r) const { return data_ + (size_t)r * cols_; }
    inline double &at_unchecked(int r, int c) { return data_[(size_t)r * cols_ + c]; }
    inline const double &at_unchecked(int r, int c) const { return data_[(size_t)r * cols_ + c]; }

    // Доступ к элементам
    double &operator()(int row, int col);
    const double &operator()(int row, int col) const;

    // Получение размеров
    int getRows() const;
    int getCols() const;

    // Операции над матрицами
    Matrix operator*(const Matrix &other) const;
    Matrix transpose() const;

    // Заполнение матрицы случайными значениями
    void fillRandom();

    // Вывод матрицы в консоль
    void print() const;

    // Функции LU-разложения
    void lu_simple(Matrix &P, Matrix &L, Matrix &U) const;
    void lu_blocked(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const;
    void lu_blocked_parallel(Matrix &P, Matrix &L, Matrix &U, int mb, int nb) const;

    // Функции разложения Холецкого
    void cholesky(Matrix &L) const;
    void cholesky_blocked(Matrix &L, int bs = 48) const;
    void cholesky_blocked_parallel(Matrix &L, int bs = 48) const;

    // Доступ к данным (для OpenBLAS и оптимизаций)
    double *data() { return data_; }
    const double *data() const { return data_; }

    // Размер данных в элементах
    size_t size() const { return data_size_; }

private:
    int rows_;
    int cols_;
    double *data_;
    size_t data_size_;
    bool owns_data_; // флаг, владеет ли эта матрица памятью
};

#endif // MATRIX_H