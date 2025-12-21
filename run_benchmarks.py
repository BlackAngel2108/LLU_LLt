import subprocess
import os
import sys

def run_benchmark():
    # --- Конфигурация ---
    # Размеры матриц для тестирования
    matrix_sizes = [256, 512] 
    # Размеры блоков
    block_sizes = [32, 64]
    # Количество потоков для параллельных версий
    thread_counts = [1, 2, 4, 8]
    # Количество запусков для каждого теста для поиска минимального времени
    runs_per_setting = 3
    # Имя исполняемого файла
    executable = "benchmark_runner.exe"
    # Файл для результатов
    results_file = "results.txt"

    # Проверяем, существует ли исполняемый файл
    if not os.path.exists(f"build/bin/{executable}"):
        print(f"Error: Executable not found at build/bin/{executable}")
        print("Please build the project first using CMake and your build tool.")
        sys.exit(1)

    # Очищаем файл с результатами
    if os.path.exists(results_file):
        os.remove(results_file)

    algorithms = [
        "lu_simple", "lu_blocked", "lu_parallel",
        "cholesky_simple", "cholesky_blocked", "cholesky_parallel"
    ]

    print("Starting benchmarks...")

    for size in matrix_sizes:
        for bs in block_sizes:
            for algo in algorithms:
                
                current_threads = [1]
                if "parallel" in algo:
                    current_threads = thread_counts

                for threads in current_threads:
                    if "simple" in algo and threads > 1:
                        continue # Пропускаем многопоточные запуски для простых версий

                    times = []
                    print(f"  Running: {algo}, Size: {size}, Block: {bs}, Threads: {threads}...")

                    for _ in range(runs_per_setting):
                        command = [
                            f"build/bin/{executable}",
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
                    with open(results_file, "a") as f:
                        f.write(f"{algo},{size},{threads},{min_time}\n")
    
    print(f"\nBenchmarking complete. Results saved to {results_file}")

if __name__ == "__main__":
    run_benchmark()
