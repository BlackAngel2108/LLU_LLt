import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_and_prepare_data():
    """Загружает и подготавливает данные с блоками"""
    
    print("Загружаем данные...")
    
    # Читаем файл
    try:
        # Загружаем данные - формат: Algorithm,MatrixSize,Threads,Block,MinTime_ms
        df = pd.read_csv("results.txt", 
                         names=['Algorithm', 'MatrixSize', 'Threads', 'Block', 'MinTime_ms'])
        
        print(f"Загружено {len(df)} строк")
        print("\nПервые 10 строк данных:")
        print(df.head(10))
        
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        
        # Попробуем загрузить вручную
        data = []
        with open("results.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('Algorithm'):  # Пропускаем заголовок если есть
                    parts = line.split(',')
                    if len(parts) == 5:
                        try:
                            data.append([
                                parts[0].strip(),                    # Algorithm
                                int(parts[1].strip()),               # MatrixSize
                                int(parts[2].strip()),               # Threads
                                int(parts[3].strip()),               # Block
                                float(parts[4].strip())              # MinTime_ms
                            ])
                        except:
                            continue
        
        if not data:
            print("Не удалось загрузить данные")
            return None
        
        df = pd.DataFrame(data, columns=['Algorithm', 'MatrixSize', 'Threads', 'Block', 'MinTime_ms'])
    
    # Удаляем строки с заголовком, если они есть
    df = df[~df['Algorithm'].astype(str).str.contains('Algorithm', case=False, na=False)]
    
    # Преобразуем типы данных на всякий случай
    df['MatrixSize'] = pd.to_numeric(df['MatrixSize'], errors='coerce')
    df['Threads'] = pd.to_numeric(df['Threads'], errors='coerce')
    df['Block'] = pd.to_numeric(df['Block'], errors='coerce')
    df['MinTime_ms'] = pd.to_numeric(df['MinTime_ms'], errors='coerce')
    
    # Удаляем NaN значения
    df = df.dropna()
    
    # Преобразуем в целые числа
    df['MatrixSize'] = df['MatrixSize'].astype(int)
    df['Threads'] = df['Threads'].astype(int)
    df['Block'] = df['Block'].astype(int)
    
    print("\n" + "="*60)
    print("ИНФОРМАЦИЯ О ДАННЫХ:")
    print("="*60)
    print(f"Всего записей: {len(df)}")
    print(f"\nУникальные алгоритмы: {df['Algorithm'].unique()}")
    print(f"Уникальные размеры матриц: {sorted(df['MatrixSize'].unique())}")
    print(f"Уникальные блоки: {sorted(df['Block'].unique())}")
    print(f"Уникальные потоки: {sorted(df['Threads'].unique())}")
    
    # Добавляем информацию об алгоритмах
    df['AlgoType'] = df['Algorithm'].apply(lambda x: 'LU' if 'lu' in str(x).lower() else 'Cholesky')
    df['IsParallel'] = df['Algorithm'].apply(lambda x: 'parallel' in str(x).lower())
    df['IsBlocked'] = df['Algorithm'].apply(lambda x: 'blocked' in str(x).lower())
    df['IsSimple'] = df['Algorithm'].apply(lambda x: 'simple' in str(x).lower())
    
    # Добавляем отображаемое имя
    df['DisplayName'] = df.apply(lambda row: f"{row['Algorithm']} (блок={row['Block']})", axis=1)
    
    print("\nТипы алгоритмов:")
    print(df[['Algorithm', 'AlgoType', 'IsParallel', 'IsBlocked']].drop_duplicates().to_string(index=False))
    
    return df

def plot_block_analysis_basic(df):
    """Основные графики анализа блоков"""
    
    output_dir = "block_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. ГРАФИК: Время vs Размер блока для разных алгоритмов
    print("\nСтрою графики анализа блоков...")
    
    # Получаем уникальные размеры блоков
    block_sizes = sorted(df['Block'].unique())
    matrix_sizes = sorted(df['MatrixSize'].unique())
    
    print(f"Размеры блоков для анализа: {block_sizes}")
    print(f"Размеры матриц: {matrix_sizes}")
    
    # Для каждого размера матрицы
    for size in matrix_sizes:
        df_size = df[df['MatrixSize'] == size]
        
        # 1.1 LU алгоритмы
        lu_df = df_size[df_size['AlgoType'] == 'LU']
        if not lu_df.empty:
            plt.figure(figsize=(14, 8))
            
            # Группируем по алгоритму и блоку
            for algo in sorted(lu_df['Algorithm'].unique()):
                algo_data = lu_df[lu_df['Algorithm'] == algo]
                
                # Для каждого количества потоков
                for threads in sorted(algo_data['Threads'].unique()):
                    thread_data = algo_data[algo_data['Threads'] == threads]
                    thread_data = thread_data.sort_values('Block')
                    
                    if not thread_data.empty:
                        label = f"{algo} - {threads} поток"
                        if threads > 1:
                            label = f"{algo} - {threads} потока"
                        
                        plt.plot(thread_data['Block'], thread_data['MinTime_ms'],
                                marker='o', linewidth=2, markersize=8, label=label)
            
            plt.title(f'LU разложение: Влияние размера блока (Матрица {size}×{size})', fontsize=14)
            plt.xlabel('Размер блока', fontsize=12)
            plt.ylabel('Время (мс)', fontsize=12)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Настраиваем ось X
            if len(block_sizes) > 1:
                plt.xticks(block_sizes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'LU_block_analysis_{size}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Сохранен: LU_block_analysis_{size}.png")
        
        # 1.2 Cholesky алгоритмы
        chol_df = df_size[df_size['AlgoType'] == 'Cholesky']
        if not chol_df.empty:
            plt.figure(figsize=(14, 8))
            
            for algo in sorted(chol_df['Algorithm'].unique()):
                algo_data = chol_df[chol_df['Algorithm'] == algo]
                
                for threads in sorted(algo_data['Threads'].unique()):
                    thread_data = algo_data[algo_data['Threads'] == threads]
                    thread_data = thread_data.sort_values('Block')
                    
                    if not thread_data.empty:
                        label = f"{algo} - {threads} поток"
                        if threads > 1:
                            label = f"{algo} - {threads} потока"
                        
                        plt.plot(thread_data['Block'], thread_data['MinTime_ms'],
                                marker='s', linewidth=2, markersize=8, label=label)
            
            plt.title(f'Cholesky разложение: Влияние размера блока (Матрица {size}×{size})', fontsize=14)
            plt.xlabel('Размер блока', fontsize=12)
            plt.ylabel('Время (мс)', fontsize=12)
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            if len(block_sizes) > 1:
                plt.xticks(block_sizes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'Cholesky_block_analysis_{size}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Сохранен: Cholesky_block_analysis_{size}.png")
    
    # 2. ГРАФИК: Сравнение производительности для разных блоков (все алгоритмы)
    for block in block_sizes:
        df_block = df[df['Block'] == block]
        
        if not df_block.empty:
            plt.figure(figsize=(14, 8))
            
            # Группируем по алгоритму и размеру матрицы
            for algo in sorted(df_block['Algorithm'].unique()):
                algo_data = df_block[df_block['Algorithm'] == algo]
                
                # Для каждого количества потоков
                for threads in sorted(algo_data['Threads'].unique()):
                    thread_data = algo_data[algo_data['Threads'] == threads]
                    thread_data = thread_data.sort_values('MatrixSize')
                    
                    if not thread_data.empty:
                        label = f"{algo} - {threads} поток"
                        if threads > 1:
                            label = f"{algo} - {threads} потока"
                        
                        plt.plot(thread_data['MatrixSize'], thread_data['MinTime_ms'],
                                marker='o', linewidth=2, markersize=8, label=label)
            
            plt.title(f'Сравнение алгоритмов (Блок = {block})', fontsize=14)
            plt.xlabel('Размер матрицы', fontsize=12)
            plt.ylabel('Время (мс)', fontsize=12)
            plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xscale('log', base=2)
            plt.yscale('log')
            plt.xticks(matrix_sizes, labels=[str(s) for s in matrix_sizes])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'all_algorithms_block_{block}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Сохранен: all_algorithms_block_{block}.png")
    
    # 3. ГРАФИК: Оптимальный размер блока для каждого алгоритма
    plt.figure(figsize=(14, 10))
    
    algorithms = sorted(df['Algorithm'].unique())
    
    for i, algo in enumerate(algorithms, 1):
        plt.subplot(2, 3, i) if len(algorithms) <= 6 else plt.subplot(3, 3, i)
        
        algo_df = df[df['Algorithm'] == algo]
        
        # Находим оптимальный блок для каждой комбинации
        optimal_points = []
        
        for size in sorted(algo_df['MatrixSize'].unique()):
            size_df = algo_df[algo_df['MatrixSize'] == size]
            
            for threads in sorted(size_df['Threads'].unique()):
                thread_df = size_df[size_df['Threads'] == threads]
                
                if not thread_df.empty:
                    # Находим блок с минимальным временем
                    min_idx = thread_df['MinTime_ms'].idxmin()
                    optimal_block = thread_df.loc[min_idx, 'Block']
                    min_time = thread_df.loc[min_idx, 'MinTime_ms']
                    
                    optimal_points.append({
                        'MatrixSize': size,
                        'Threads': threads,
                        'OptimalBlock': optimal_block,
                        'MinTime': min_time
                    })
        
        if optimal_points:
            opt_df = pd.DataFrame(optimal_points)
            
            # Визуализируем
            for size in sorted(opt_df['MatrixSize'].unique()):
                size_data = opt_df[opt_df['MatrixSize'] == size]
                size_data = size_data.sort_values('Threads')
                
                label = f'{size}×{size}'
                plt.plot(size_data['Threads'], size_data['OptimalBlock'],
                        marker='o', linewidth=2, markersize=6, label=label)
        
        plt.title(f'{algo}\nОптимальный блок', fontsize=11)
        plt.xlabel('Потоки')
        plt.ylabel('Лучший блок')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.xticks([1, 2, 4, 8])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_blocks_per_algorithm.png'), dpi=150)
    plt.close()
    print(f"  Сохранен: optimal_blocks_per_algorithm.png")
    
    return output_dir

def plot_block_vs_threads_heatmaps(df):
    """Heatmap: Блоки vs Потоки"""
    
    output_dir = "heatmaps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sns.set_theme(style="whitegrid")
    
    # Для каждого алгоритма с блоками
    blocked_algos = [a for a in df['Algorithm'].unique() if 'blocked' in a]
    
    for algo in blocked_algos:
        algo_df = df[df['Algorithm'] == algo]
        
        if not algo_df.empty:
            # Для каждого размера матрицы
            for size in sorted(algo_df['MatrixSize'].unique()):
                size_df = algo_df[algo_df['MatrixSize'] == size]
                
                # Создаем pivot таблицу
                pivot_data = size_df.pivot_table(
                    values='MinTime_ms',
                    index='Block',
                    columns='Threads',
                    aggfunc='mean'
                )
                
                # Сортируем
                pivot_data = pivot_data.sort_index()
                pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
                
                if not pivot_data.empty:
                    plt.figure(figsize=(10, 8))
                    
                    sns.heatmap(pivot_data,
                               annot=True,
                               fmt='.1f',
                               cmap='YlOrRd',
                               cbar_kws={'label': 'Время (мс)'},
                               linewidths=0.5,
                               square=True)
                    
                    plt.title(f'{algo} - {size}×{size}\nБлоки vs Потоки', fontsize=14)
                    plt.xlabel('Количество потоков', fontsize=12)
                    plt.ylabel('Размер блока', fontsize=12)
                    
                    plt.tight_layout()
                    filename = f'{algo}_{size}_block_thread_heatmap.png'
                    plt.savefig(os.path.join(output_dir, filename), dpi=150)
                    plt.close()
                    print(f"  Сохранен: {filename}")

def generate_block_recommendations(df):
    """Генерирует рекомендации по блокам"""
    
    print("\n" + "="*80)
    print("АНАЛИЗ ОПТИМАЛЬНЫХ РАЗМЕРОВ БЛОКОВ")
    print("="*80)
    
    recommendations = []
    
    for algo in df['Algorithm'].unique():
        algo_df = df[df['Algorithm'] == algo]
        
        if 'blocked' in algo:  # Только для алгоритмов с блоками
            print(f"\n{algo}:")
            
            for size in sorted(algo_df['MatrixSize'].unique()):
                size_df = algo_df[algo_df['MatrixSize'] == size]
                
                print(f"  Матрица {size}×{size}:")
                
                for threads in sorted(size_df['Threads'].unique()):
                    thread_df = size_df[size_df['Threads'] == threads]
                    
                    if not thread_df.empty:
                        # Находим лучший и худший блок
                        min_idx = thread_df['MinTime_ms'].idxmin()
                        max_idx = thread_df['MinTime_ms'].idxmax()
                        
                        best_block = thread_df.loc[min_idx, 'Block']
                        best_time = thread_df.loc[min_idx, 'MinTime_ms']
                        worst_block = thread_df.loc[max_idx, 'Block']
                        worst_time = thread_df.loc[max_idx, 'MinTime_ms']
                        
                        # Вычисляем улучшение
                        improvement = 100 * (worst_time - best_time) / best_time
                        
                        print(f"    {threads} поток(ов): лучший блок={best_block} ({best_time:.1f} мс), "
                              f"худший={worst_block} ({worst_time:.1f} мс), улучшение={improvement:.1f}%")
                        
                        if improvement > 10:  # Значимое улучшение
                            recommendations.append({
                                'Algorithm': algo,
                                'MatrixSize': size,
                                'Threads': threads,
                                'BestBlock': best_block,
                                'WorstBlock': worst_block,
                                'Improvement%': f"{improvement:.1f}",
                                'BestTime_ms': f"{best_time:.1f}"
                            })
    
    # Сохраняем рекомендации
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        rec_df.to_csv("block_recommendations.csv", index=False)
        print(f"\nРекомендации сохранены в block_recommendations.csv")
    
    return recommendations

def plot_summary_comparison(df):
    """Сводный график сравнения"""
    
    output_dir = "summary"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Сравнение всех алгоритмов для каждого размера матрицы
    for size in sorted(df['MatrixSize'].unique()):
        df_size = df[df['MatrixSize'] == size]
        
        plt.figure(figsize=(16, 10))
        
        # Создаем комбинированный идентификатор для легенды
        df_size['LegendLabel'] = df_size.apply(
            lambda row: f"{row['Algorithm']} (блок={row['Block']}, потоки={row['Threads']})", axis=1
        )
        
        # Для каждого уникального сочетания
        for label in sorted(df_size['LegendLabel'].unique()):
            label_data = df_size[df_size['LegendLabel'] == label]
            
            # Если данные для одного потока - точка
            if len(label_data) == 1:
                plt.scatter(label_data['Algorithm'], label_data['MinTime_ms'],
                           s=100, label=label, alpha=0.7)
            # Если несколько потоков - линия
            else:
                label_data = label_data.sort_values('Threads')
                plt.plot(label_data['Threads'], label_data['MinTime_ms'],
                        marker='o', linewidth=2, markersize=8, label=label)
        
        plt.title(f'Сравнение всех алгоритмов (Матрица {size}×{size})', fontsize=16)
        plt.xlabel('Алгоритм / Количество потоков', fontsize=14)
        plt.ylabel('Время (мс)', fontsize=14)
        plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'full_comparison_{size}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Сохранен: full_comparison_{size}.png")

def main():
    """Основная функция"""
    
    print("Анализ влияния размера блока на производительность")
    print("="*60)
    
    # Загружаем данные
    df = load_and_prepare_data()
    
    if df is None or len(df) == 0:
        print("Ошибка: Нет данных для анализа")
        return
    
    # Строим основные графики
    plot_dir = plot_block_analysis_basic(df)
    
    # Heatmap
    plot_block_vs_threads_heatmaps(df)
    
    # Генерируем рекомендации
    recommendations = generate_block_recommendations(df)
    
    # Сводный график
    plot_summary_comparison(df)
    
    # Выводим лучшие результаты
    print("\n" + "="*80)
    print("ЛУЧШИЕ РЕЗУЛЬТАТЫ ДЛЯ КАЖДОГО АЛГОРИТМА")
    print("="*80)
    
    best_results = []
    
    for algo in df['Algorithm'].unique():
        algo_df = df[df['Algorithm'] == algo]
        
        if not algo_df.empty:
            # Находим абсолютно лучший результат для этого алгоритма
            min_idx = algo_df['MinTime_ms'].idxmin()
            best = algo_df.loc[min_idx]
            
            best_results.append({
                'Algorithm': best['Algorithm'],
                'MatrixSize': best['MatrixSize'],
                'Threads': best['Threads'],
                'Block': best['Block'],
                'Time_ms': f"{best['MinTime_ms']:.2f}"
            })
    
    best_df = pd.DataFrame(best_results)
    print(best_df.to_string(index=False))
    
    # Сохраняем лучшие результаты
    best_df.to_csv("best_results.csv", index=False)
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("="*60)
    print("\nСозданы папки с графиками:")
    print("  - block_analysis/ - основные графики по блокам")
    print("  - heatmaps/ - heatmap анализа блоков и потоков")
    print("  - summary/ - сводные графики сравнения")
    print("\nСозданы файлы:")
    print("  - block_recommendations.csv - рекомендации по выбору блоков")
    print("  - best_results.csv - таблица лучших результатов")

if __name__ == "__main__":
    main()