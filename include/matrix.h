#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class Matrix {
public:
    // Конструктор
    Matrix(int rows, int cols);
    // Конструктор копирования
    Matrix(const Matrix& other);
    // Оператор присваивания
    Matrix& operator=(const Matrix& other);

    // Доступ к элементам
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;

    // Получение размеров
    int getRows() const;
    int getCols() const;

    // Операции над матрицами
    Matrix operator*(const Matrix& other) const;
    Matrix transpose() const;


    // Заполнение матрицы случайными значениями
    void fillRandom();

    // Вывод матрицы в консоль
    void print() const;
    
    // Функции LU-разложения
    // Возвращают пару матриц: L и U
    std::pair<Matrix, Matrix> lu_simple() const;
    std::pair<Matrix, Matrix> lu_blocked(int block_size) const;
    std::pair<Matrix, Matrix> lu_blocked_parallel(int block_size) const;

    std::pair<Matrix, Matrix> lu_simple_parallel(); // New
    // Функция разложения Холецкого (LLt)
    // Возвращает нижнетреугольную матрицу L
    Matrix cholesky() const;
    Matrix cholesky_blocked(int block_size) const;
    Matrix cholesky_blocked_parallel(int block_size) const;
    Matrix cholesky_simple_parallel(); // New

private:
    int rows_;
    int cols_;
    std::vector<double> data_;
};

#endif // MATRIX_H
