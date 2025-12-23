import subprocess
import os
import sys

def run_benchmark():
    # --- Конфигурация ---
    # Размеры матриц для тестирования
    matrix_sizes = [1024,2048] 
    # Размеры блоков
    block_sizes = [32, 64]
    # Количество потоков для параллельных версий
    thread_counts = [1, 2, 4, 8]
    # Количество запусков для каждого теста для поиска минимального времени
    runs_per_setting = 3
    # Имя исполняемого файла
    executable_path = "LLU_LLt/build/bin/benchmark_runner.exe" # Updated path
    # Файл для результатов
    results_file = "results.txt"

    # Проверяем, существует ли исполняемый файл
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        print("Please build the project first using CMake and your build tool.")
        sys.exit(1)

    # Очищаем файл с результатами
    if os.path.exists(results_file):
        os.remove(results_file)

    algorithms = [
        "lu_simple", "lu_blocked", "lu_blocked_parallel", # Corrected algorithm names
        "cholesky_simple", "cholesky_blocked", "cholesky_blocked_parallel" # Corrected algorithm names
    ]

    print("Starting benchmarks...")

    with open(results_file, "w") as f_results: # Open in write mode to create/clear header
        f_results.write("Algorithm,MatrixSize,Threads,MinTime_ms\n")

    for size in matrix_sizes:
        for bs in block_sizes:
            for algo in algorithms:
                
                current_threads = [1]
                # Use only thread_counts for parallel versions
                if "parallel" in algo:
                    current_threads = thread_counts
                elif "blocked" in algo and "lu" in algo: # For non-parallel blocked LU, still use various threads to see if any implicit threading happens
                    current_threads = thread_counts
                
                for threads in current_threads:
                    if "simple" in algo and threads > 1:
                        continue # Пропускаем многопоточные запуски для простых версий

                    times = []
                    print(f"  Running: {algo}, Size: {size}, Block: {bs}, Threads: {threads}...")

                    for _ in range(runs_per_setting):
                        command = [
                            executable_path, # Use the explicit path
                            algo,
                            str(size),
                            str(bs),
                            str(threads)
                        ]
                        
                        result = subprocess.run(command, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            print(f"    Error running benchmark: {result.stderr}")
                            continue
                        
                        try:
                            # Последняя строка вывода должна быть в формате "algo,size,threads,time"
                            output_line = result.stdout.strip().splitlines()[-1]
                            time_ms = float(output_line.split(',')[-1])
                            times.append(time_ms)
                        except (IndexError, ValueError) as e:
                            print(f"    Could not parse output: {result.stdout}. Error: {e}")

                    if not times:
                        print("    No successful runs for this setting.")
                        continue

                    min_time = min(times)
                    print(f"    Min time over {runs_per_setting} runs: {min_time:.4f} ms")

                    # Записываем лучший результат
                    with open(results_file, "a") as f_results:
                        f_results.write(f"{algo},{size},{threads},{min_time}\n")
    
    print(f"\nBenchmarking complete. Results saved to {results_file}")

if __name__ == "__main__":
    run_benchmark()