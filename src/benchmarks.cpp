#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <omp.h>
#include "matrix.h"
//#include <cblas.h>
//#include <lapacke.h>
#include <unistd.h>
#include <tuple>
#include <stdexcept>

// Структура для алгоритма
struct Algorithm
{
    std::string name;
    bool is_parallel;
    bool is_blocked;
    bool use_openblas;
};

// Структура для результата сравнения
struct ComparisonResult
{
    std::string algorithm;
    int size;
    int block;
    int threads;
    double my_time;
    double openblas_time;
    double difference;
    double speedup;
};

// Функция для LU разложения через OpenBLAS
void openblas_lu(Matrix &A)
{
    int n = A.getRows();
    //std::vector<int> ipiv(n);
    //LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A.data(), n, ipiv.data());
}

// Функция для Cholesky разложения через OpenBLAS
void openblas_cholesky(Matrix &A)
{
    int n = A.getRows();
    char uplo = 'L';
    //LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, A.data(), n);
}

// Функция запуска одного замера для моего кода
double run_my_benchmark(const std::string &type, int size, int block_size, int threads, int runs = 1)
{
    std::vector<double> times;
    times.reserve(runs);

    for (int run = 0; run < runs; ++run)
    {
        Matrix A(size, size);
        A.fillRandom();

        if (type.find("cholesky") != std::string::npos)
        {
            Matrix A_t = A.transpose();
            A = A * A_t;

            // Чтобы матрица была строго положительно определенной
            for (int i = 0; i < size; ++i)
            {
                A(i, i) += static_cast<double>(size);
            }
        }

        omp_set_num_threads(threads);

        auto start = std::chrono::high_resolution_clock::now();

        if (type == "lu_simple")
        {
            Matrix P(size, size);
            Matrix L(size, size);
            Matrix U(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.lu_simple(P, L, U);
        }
        else if (type == "lu_blocked")
        {
            Matrix P(size, size);
            Matrix L(size, size);
            Matrix U(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.lu_blocked(P, L, U, block_size, block_size);
        }
        else if (type == "lu_blocked_parallel")
        {
            Matrix P(size, size);
            Matrix L(size, size);
            Matrix U(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.lu_blocked_parallel(P, L, U, block_size, block_size);
        }

#ifdef USE_RVV
        else if (type == "lu_blocked_parallel_rvv_m1")
        {
            Matrix P(size, size);
            Matrix L(size, size);
            Matrix U(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.lu_blocked_parallel_rvv_m1(P, L, U, block_size, block_size);
        }
        else if (type == "lu_blocked_parallel_rvv_m2")
        {
            Matrix P(size, size);
            Matrix L(size, size);
            Matrix U(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.lu_blocked_parallel_rvv_m2(P, L, U, block_size, block_size);
        }
        else if (type == "lu_blocked_parallel_rvv_m4")
        {
            Matrix P(size, size);
            Matrix L(size, size);
            Matrix U(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.lu_blocked_parallel_rvv_m4(P, L, U, block_size, block_size);
        }
        else if (type == "lu_blocked_parallel_rvv_m8")
        {
            Matrix P(size, size);
            Matrix L(size, size);
            Matrix U(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.lu_blocked_parallel_rvv_m8(P, L, U, block_size, block_size);
        }
#endif

        else if (type == "cholesky_simple")
        {
            Matrix L(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.cholesky(L);
        }
        else if (type == "cholesky_blocked")
        {
            Matrix L(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.cholesky_blocked(L, block_size);
        }
        else if (type == "cholesky_blocked_parallel")
        {
            Matrix L(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.cholesky_blocked_parallel(L, block_size);
        }

#ifdef USE_RVV
        else if (type == "cholesky_blocked_parallel_rvv_m1")
        {
            Matrix L(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.cholesky_blocked_parallel_rvv_m1(L, block_size);
        }
        else if (type == "cholesky_blocked_parallel_rvv_m2")
        {
            Matrix L(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.cholesky_blocked_parallel_rvv_m2(L, block_size);
        }
        else if (type == "cholesky_blocked_parallel_rvv_m4")
        {
            Matrix L(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.cholesky_blocked_parallel_rvv_m4(L, block_size);
        }
        else if (type == "cholesky_blocked_parallel_rvv_m8")
        {
            Matrix L(size, size);

            start = std::chrono::high_resolution_clock::now();
            A.cholesky_blocked_parallel_rvv_m8(L, block_size);
        }
#endif

        else
        {
            throw std::runtime_error("Unknown benchmark type: " + type);
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
    }

    return *std::min_element(times.begin(), times.end());
}

// Функция запуска одного замера для OpenBLAS
double run_openblas_benchmark(const std::string &type, int size, int threads, int runs = 1)
{
    std::vector<double> times;
    times.reserve(runs);

    for (int run = 0; run < runs; ++run)
    {
        Matrix A(size, size);
        A.fillRandom();

        if (type.find("cholesky") != std::string::npos)
        {
            Matrix A_t = A.transpose();
            A = A * A_t;

            for (int i = 0; i < size; ++i)
            {
                A(i, i) += static_cast<double>(size);
            }
        }

        //openblas_set_num_threads(threads);

        auto start = std::chrono::high_resolution_clock::now();

        // if (type.find("lu") != std::string::npos)
        // {
        //     openblas_lu(A);
        // }
        // else if (type.find("cholesky") != std::string::npos)
        // {
        //     openblas_cholesky(A);
        // }

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
    }

    return *std::min_element(times.begin(), times.end());
}

// Функция для запуска бенчмарков моего кода
void run_my_benchmarks(const std::string &results_file)
{
    std::vector<int> matrix_sizes = {1000, 2000, 4000, 6000, 8000};
    std::vector<int> block_sizes = {50, 100, 150};
    std::vector<int> thread_counts = {1, 2, 3, 4, 5, 6, 7, 8};
    const int runs_per_setting = 3;

    std::vector<Algorithm> algorithms = {
        {"lu_simple", false, false, false},
        {"lu_blocked", false, true, false},
        {"lu_blocked_parallel", true, true, false},
        {"cholesky_simple", false, false, false},
        {"cholesky_blocked", false, true, false},
        {"cholesky_blocked_parallel", true, true, false}
    };

    std::ofstream f_results(results_file);
    f_results << "Algorithm,MatrixSize,Threads,Block,MyTime_ms\n";
    f_results.close();

    std::cout << "\nRunning my benchmarks...\n";

    for (int size : matrix_sizes)
    {
        for (const auto &algo : algorithms)
        {
            if (!algo.is_blocked && !algo.is_parallel)
            {
                // Простые алгоритмы
                std::cout << "  " << algo.name << ", size=" << size << std::endl;

                double min_time = run_my_benchmark(algo.name, size, 0, 1, runs_per_setting);

                std::ofstream f_results_append(results_file, std::ios::app);
                f_results_append << algo.name << "," << size << ",1,0,"
                                 << std::fixed << std::setprecision(4) << min_time << "\n";
                f_results_append.close();
            }
            else if (algo.is_blocked && !algo.is_parallel)
            {
                // Блочные непараллельные
                for (int bs : block_sizes)
                {
                    std::cout << "  " << algo.name
                              << ", size=" << size
                              << ", block=" << bs
                              << std::endl;

                    double min_time = run_my_benchmark(algo.name, size, bs, 1, runs_per_setting);

                    std::ofstream f_results_append(results_file, std::ios::app);
                    f_results_append << algo.name << "," << size << ",1," << bs << ","
                                     << std::fixed << std::setprecision(4) << min_time << "\n";
                    f_results_append.close();
                }
            }
            else if (algo.is_blocked && algo.is_parallel)
            {
                // Блочные параллельные
                for (int bs : block_sizes)
                {
                    for (int threads : thread_counts)
                    {
                        std::cout << "  " << algo.name
                                  << ", size=" << size
                                  << ", block=" << bs
                                  << ", threads=" << threads
                                  << std::endl;

                        double min_time = run_my_benchmark(algo.name, size, bs, threads, runs_per_setting);

                        std::ofstream f_results_append(results_file, std::ios::app);
                        f_results_append << algo.name << "," << size << "," << threads << "," << bs << ","
                                         << std::fixed << std::setprecision(4) << min_time << "\n";
                        f_results_append.close();
                    }
                }
            }
        }
    }
}

// Функция для запуска RVV LMUL бенчмарков
void run_rvv_benchmarks(const std::string &results_file)
{
#ifndef USE_RVV
    std::cerr << "RVV benchmarks are disabled. Build with -DUSE_RVV and RISC-V vector flags.\n";
    return;
#else
    std::vector<int> matrix_sizes = {1000, 2000, 4000, 6000, 8000};
    std::vector<int> block_sizes = {50, 100, 150};
    std::vector<int> thread_counts = {1, 2, 3, 4, 5, 6, 7, 8};
    const int runs_per_setting = 3;

    struct RvvAlgorithm
    {
        std::string name;
        std::string base_method;
        std::string lmul;
    };

    std::vector<RvvAlgorithm> algorithms = {
        {"lu_blocked_parallel_rvv_m1", "LU", "m1"},
        {"lu_blocked_parallel_rvv_m2", "LU", "m2"},
        {"lu_blocked_parallel_rvv_m4", "LU", "m4"},
        {"lu_blocked_parallel_rvv_m8", "LU", "m8"},

        {"cholesky_blocked_parallel_rvv_m1", "Cholesky", "m1"},
        {"cholesky_blocked_parallel_rvv_m2", "Cholesky", "m2"},
        {"cholesky_blocked_parallel_rvv_m4", "Cholesky", "m4"},
        {"cholesky_blocked_parallel_rvv_m8", "Cholesky", "m8"}
    };

    std::ofstream f_results(results_file);

    if (!f_results.is_open())
    {
        std::cerr << "ERROR: Cannot open file " << results_file << " for writing!" << std::endl;
        return;
    }

    f_results << "Algorithm,BaseMethod,LMUL,MatrixSize,Threads,Block,Time_ms\n";
    f_results.close();

    std::cout << "\nRunning RVV LMUL benchmarks...\n";

    for (int size : matrix_sizes)
    {
        for (const auto &algo : algorithms)
        {
            for (int bs : block_sizes)
            {
                for (int threads : thread_counts)
                {
                    std::cout << "  " << algo.name
                              << ", method=" << algo.base_method
                              << ", LMUL=" << algo.lmul
                              << ", size=" << size
                              << ", block=" << bs
                              << ", threads=" << threads
                              << std::endl;

                    double min_time = run_my_benchmark(
                        algo.name,
                        size,
                        bs,
                        threads,
                        runs_per_setting
                    );

                    std::ofstream f_results_append(results_file, std::ios::app);

                    if (!f_results_append.is_open())
                    {
                        std::cerr << "ERROR: Cannot open file " << results_file << " for appending!" << std::endl;
                        return;
                    }

                    f_results_append << algo.name << ","
                                     << algo.base_method << ","
                                     << algo.lmul << ","
                                     << size << ","
                                     << threads << ","
                                     << bs << ","
                                     << std::fixed << std::setprecision(4)
                                     << min_time << "\n";

                    f_results_append.close();
                }
            }
        }
    }

    std::cout << "\nRVV benchmarks complete. Results saved to "
              << results_file << std::endl;
#endif
}

// Функция для запуска бенчмарков OpenBLAS
void run_openblas_benchmarks(const std::string &results_file)
{
    std::vector<int> matrix_sizes = {256, 512, 1024};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    const int runs_per_setting = 3;

    std::vector<std::string> algorithms = {
        "lu_simple",
        "lu_blocked",
        "lu_blocked_parallel",
        "cholesky_simple",
        "cholesky_blocked",
        "cholesky_blocked_parallel"
    };

    std::ofstream f_results(results_file);
    f_results << "Algorithm,MatrixSize,Threads,OpenBLAS_Time_ms\n";
    f_results.close();

    std::cout << "\nRunning OpenBLAS benchmarks...\n";

    for (int size : matrix_sizes)
    {
        for (const auto &algo : algorithms)
        {
            for (int threads : thread_counts)
            {
                std::cout << "  " << algo
                          << ", size=" << size
                          << ", threads=" << threads
                          << std::endl;

                double min_time = run_openblas_benchmark(algo, size, threads, runs_per_setting);

                std::ofstream f_results_append(results_file, std::ios::app);
                f_results_append << algo << "," << size << "," << threads << ","
                                 << std::fixed << std::setprecision(4) << min_time << "\n";
                f_results_append.close();
            }
        }
    }
}

// Сравнение путем запуска обоих алгоритмов
void compare_algorithms(const std::string &comparison_file)
{
    char cwd[1024];

    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        std::cout << "Current working directory: " << cwd << std::endl;
        std::cout << "Looking for file: " << cwd << "/" << comparison_file << std::endl;
    }

    std::vector<int> matrix_sizes = {1000, 2000, 3000, 4000};
    std::vector<int> block_sizes = {32};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    const int runs_per_setting = 3;

    std::vector<Algorithm> algorithms = {
        {"lu_simple", false, false, false},
        {"lu_blocked", false, true, false},
        {"lu_blocked_parallel", true, true, false},
        {"cholesky_simple", false, false, false},
        {"cholesky_blocked", false, true, false},
        {"cholesky_blocked_parallel", true, true, false}
    };

    std::ofstream comp_out(comparison_file);

    if (!comp_out.is_open())
    {
        std::cerr << "ERROR: Cannot open file " << comparison_file << " for writing!" << std::endl;
        return;
    }

    comp_out << "Algorithm,MatrixSize,Threads,Block,MyTime_ms,OpenBLAS_Time_ms,Difference_ms,Speedup\n";
    comp_out << std::flush;

    std::cout << "\nRunning comparison benchmarks...\n";

    for (int size : matrix_sizes)
    {
        for (const auto &algo : algorithms)
        {
            if (!algo.is_blocked && !algo.is_parallel)
            {
                // Простые алгоритмы
                std::cout << "  Comparing " << algo.name
                          << ", size=" << size
                          << std::endl;

                double my_time = run_my_benchmark(algo.name, size, 0, 1, runs_per_setting);
                double openblas_time = run_openblas_benchmark(algo.name, size, 1, runs_per_setting);

                double difference = my_time - openblas_time;
                double speedup = openblas_time / my_time;

                std::cout << algo.name << "," << size << ",1,0,"
                          << std::fixed << std::setprecision(4) << my_time << ","
                          << std::fixed << std::setprecision(4) << openblas_time << ","
                          << std::fixed << std::setprecision(4) << difference << ","
                          << std::fixed << std::setprecision(4) << speedup << "\n";

                comp_out << algo.name << "," << size << ",1,0,"
                         << std::fixed << std::setprecision(4) << my_time << ","
                         << std::fixed << std::setprecision(4) << openblas_time << ","
                         << std::fixed << std::setprecision(4) << difference << ","
                         << std::fixed << std::setprecision(4) << speedup << "\n";

                comp_out << std::flush;
            }
            else if (algo.is_blocked && !algo.is_parallel)
            {
                // Блочные непараллельные
                for (int bs : block_sizes)
                {
                    std::cout << "  Comparing " << algo.name
                              << ", size=" << size
                              << ", block=" << bs
                              << std::endl;

                    double my_time = run_my_benchmark(algo.name, size, bs, 1, runs_per_setting);
                    double openblas_time = run_openblas_benchmark(algo.name, size, 1, runs_per_setting);

                    double difference = my_time - openblas_time;
                    double speedup = openblas_time / my_time;

                    comp_out << algo.name << "," << size << ",1," << bs << ","
                             << std::fixed << std::setprecision(4) << my_time << ","
                             << std::fixed << std::setprecision(4) << openblas_time << ","
                             << std::fixed << std::setprecision(4) << difference << ","
                             << std::fixed << std::setprecision(4) << speedup << "\n";

                    comp_out << std::flush;

                    std::cout << algo.name << "," << size << ",1," << bs << ","
                              << std::fixed << std::setprecision(4) << my_time << ","
                              << std::fixed << std::setprecision(4) << openblas_time << ","
                              << std::fixed << std::setprecision(4) << difference << ","
                              << std::fixed << std::setprecision(4) << speedup << "\n";
                }
            }
            else if (algo.is_blocked && algo.is_parallel)
            {
                // Блочные параллельные
                for (int bs : block_sizes)
                {
                    for (int threads : thread_counts)
                    {
                        std::cout << "  Comparing " << algo.name
                                  << ", size=" << size
                                  << ", block=" << bs
                                  << ", threads=" << threads
                                  << std::endl;

                        double my_time = run_my_benchmark(algo.name, size, bs, threads, runs_per_setting);
                        double openblas_time = run_openblas_benchmark(algo.name, size, threads, runs_per_setting);

                        double difference = my_time - openblas_time;
                        double speedup = openblas_time / my_time;

                        comp_out << algo.name << "," << size << "," << threads << "," << bs << ","
                                 << std::fixed << std::setprecision(4) << my_time << ","
                                 << std::fixed << std::setprecision(4) << openblas_time << ","
                                 << std::fixed << std::setprecision(4) << difference << ","
                                 << std::fixed << std::setprecision(4) << speedup << "\n";

                        comp_out << std::flush;

                        std::cout << algo.name << "," << size << "," << threads << "," << bs << ","
                                  << std::fixed << std::setprecision(4) << my_time << ","
                                  << std::fixed << std::setprecision(4) << openblas_time << ","
                                  << std::fixed << std::setprecision(4) << difference << ","
                                  << std::fixed << std::setprecision(4) << speedup << "\n";
                    }
                }
            }
        }
    }

    comp_out.close();

    std::cout << "\nComparison complete. Results saved to "
              << comparison_file << std::endl;
}

// Функция для печати справки
void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " <mode>\n"
              << "Modes:\n"
              << "  my         - Run my benchmarks (saves to my_results.txt)\n"
              << "  openblas   - Run OpenBLAS benchmarks (saves to openblas_results.txt)\n"
              << "  both       - Run both separately (saves to separate files)\n"
              << "  compare    - Run comparison (saves to comparison_results.txt)\n"
              << "  rvv        - Run RVV LMUL benchmarks (saves to rvv_results.txt)\n";
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "my")
    {
        run_my_benchmarks("my_results.txt");
        std::cout << "\nMy benchmarks complete. Results saved to my_results.txt" << std::endl;
    }
    else if (mode == "openblas")
    {
        run_openblas_benchmarks("openblas_results.txt");
        std::cout << "\nOpenBLAS benchmarks complete. Results saved to openblas_results.txt" << std::endl;
    }
    else if (mode == "both")
    {
        run_my_benchmarks("my_results.txt");
        run_openblas_benchmarks("openblas_results.txt");
        std::cout << "\nBoth benchmarks complete. Results saved to separate files." << std::endl;
    }
    else if (mode == "compare")
    {
        compare_algorithms("comparison_results.txt");
    }
    else if (mode == "rvv")
    {
        run_rvv_benchmarks("rvv_results.txt");
    }
    else
    {
        std::cerr << "Unknown mode: " << mode << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}