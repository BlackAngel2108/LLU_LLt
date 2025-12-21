import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results():
    results_file = "results.txt"
    output_dir = "plots"

    # Проверяем, существует ли файл с результатами
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at '{results_file}'")
        print("Please run 'run_benchmarks.py' first.")
        return

    # Создаем папку для графиков
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Загружаем данные
    try:
        df = pd.read_csv(results_file, names=['algorithm', 'size', 'threads', 'time'])
    except pd.errors.EmptyDataError:
        print(f"Error: The results file '{results_file}' is empty.")
        return

    # Устанавливаем стиль графиков
    sns.set_theme(style="whitegrid")

    # Получаем уникальные размеры матриц из результатов
    matrix_sizes = df['size'].unique()

    for size in matrix_sizes:
        df_size = df[df['size'] == size]

        # --- График для LU разложений ---
        plt.figure(figsize=(12, 8))
        lu_df = df_size[df_size['algorithm'].str.contains('lu')]
        
        # Простые и блочные алгоритмы (отображаются как горизонтальные линии)
        for algo_type in ['lu_simple', 'lu_blocked']:
            data = lu_df[lu_df['algorithm'] == algo_type]
            if not data.empty:
                time = data['time'].iloc[0]
                plt.axhline(y=time, linestyle='--', label=f'{algo_type} (1 thread)')
        
        # Параллельный алгоритм
        parallel_data = lu_df[lu_df['algorithm'] == 'lu_parallel']
        if not parallel_data.empty:
            sns.lineplot(data=parallel_data, x='threads', y='time', marker='o', label='lu_parallel')

        plt.title(f'Производительность LU разложения (Матрица {size}x{size})')
        plt.xlabel('Количество потоков')
        plt.ylabel('Время (мс)')
        plt.legend()
        plt.xscale('log', base=2) # Логарифмическая шкала для потоков
        plt.xticks(parallel_data['threads'].unique())
        lu_plot_path = os.path.join(output_dir, f'LU_performance_{size}x{size}.png')
        plt.savefig(lu_plot_path)
        plt.close()
        print(f"Saved LU plot to {lu_plot_path}")

        # --- График для LLt (Cholesky) разложений ---
        plt.figure(figsize=(12, 8))
        cholesky_df = df_size[df_size['algorithm'].str.contains('cholesky')]

        # Простые и блочные алгоритмы
        for algo_type in ['cholesky_simple', 'cholesky_blocked']:
            data = cholesky_df[cholesky_df['algorithm'] == algo_type]
            if not data.empty:
                time = data['time'].iloc[0]
                plt.axhline(y=time, linestyle='--', label=f'{algo_type} (1 thread)')

        # Параллельный алгоритм
        parallel_data = cholesky_df[cholesky_df['algorithm'] == 'cholesky_parallel']
        if not parallel_data.empty:
            sns.lineplot(data=parallel_data, x='threads', y='time', marker='o', label='cholesky_parallel')

        plt.title(f'Производительность LLt разложения (Матрица {size}x{size})')
        plt.xlabel('Количество потоков')
        plt.ylabel('Время (мс)')
        plt.legend()
        plt.xscale('log', base=2)
        plt.xticks(parallel_data['threads'].unique())
        cholesky_plot_path = os.path.join(output_dir, f'Cholesky_performance_{size}x{size}.png')
        plt.savefig(cholesky_plot_path)
        plt.close()
        print(f"Saved Cholesky plot to {cholesky_plot_path}")

if __name__ == "__main__":
    plot_results()
