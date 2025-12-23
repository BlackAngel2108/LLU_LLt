#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "matrix.h"
#include <omp.h>

void run_decomposition(const std::string& type, int size, int block_size, int threads) {
    Matrix A(size, size);
    A.fillRandom();

    // Для симметричных матриц (нужно для Холецкого)
    if (type.find("cholesky") != std::string::npos) {
        Matrix A_t = A.transpose();
        A = A * A_t; 
    }

    omp_set_num_threads(threads);

    auto start = std::chrono::high_resolution_clock::now();

    if (type == "lu_simple") {
        A.lu_simple();
    } else if (type == "lu_blocked") {
        A.lu_blocked(block_size);
    } else if (type == "lu_blocked_parallel") {
        A.lu_blocked_parallel(block_size);
    } else if (type == "cholesky_simple") {
        A.cholesky();
    } else if (type == "cholesky_blocked") {
        A.cholesky_blocked(block_size);
    } else if (type == "cholesky_blocked_parallel") {
        A.cholesky_blocked_parallel(block_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Вывод в формате: АЛГОРИТМ,РАЗМЕР,ПОТОКИ,ВРЕМЯ_мс
    std::cout << type << "," << size << "," << threads << "," << duration.count() << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <type> <matrix_size> <block_size> <num_threads>" << std::endl;
        return 1;
    }

    try {
        std::string type = argv[1];
        int matrix_size = std::stoi(argv[2]);
        int block_size = std::stoi(argv[3]);
        int num_threads = std::stoi(argv[4]);

        run_decomposition(type, matrix_size, block_size, num_threads);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}