import pandas as pd
import json
import numpy as np
import os

# пути
input_path = 'data/processed/datasets/final_dataset.csv'
output_path = 'data/processed/datasets/vectorized_dataset.csv'
ranges_path = 'data/references/ranges.json'

def runnormalization():
    # 1. загружаем данные
    print("загружаю данные...")
    df = pd.read_csv(input_path)
    with open(ranges_path, 'r') as f:
        ranges = json.load(f)

    # 2. функция-калькулятор 
    def calculatemetrics(row, col_name):
        val = row[col_name]
        if pd.isna(val): 
            return pd.Series([np.nan, np.nan])

        # вытягиваем нормы из джейсона
        info = ranges[col_name]
        if info['sex_dependent']:
            gender = 'male' if row['demo_RIAGENDR'] == 1.0 else 'female'
            r = info['ranges'][gender]
        else:
            r = info['ranges']['any']
        
        min_n, max_n = r['min'], r['max']
        
        width = max_n - min_n
        center = (max_n + min_n) / 2
        
        # 1. считаю dev_vector (отклонение от центра)
        # если ширина 1.9, а ты выше центра на 0.5, то отклонение будет ~0.26
        deviation = (val - center) / width
        
        # 2. считаю med_status (твой "индикатор пиздеца")
        abs_dev = abs(deviation)
        sign = 1 if deviation >= 0 else -1 # сохраняем знак: плюс или минус
        
        # логика "серой зоны" (15% от краев):
        # 0.35 — это 70% центральной зоны (от -0.35 до +0.35)
        # 0.5 — это граница нормы
        if abs_dev <= 0.35:
            status = 0 # идеальное здоровье
        elif abs_dev <= 0.5:
            status = 1 * sign # серая зона (погранично)
        elif abs_dev <= 1.0:
            status = 2 * sign # вылет за норму
        else:
            status = 3 * sign # критический вылет
            
        return pd.Series([deviation, status])
        

    # 3. цикл по всем колонкам из джейсона
    for col in ranges.keys():
        if col in df.columns:
            print(f"обрабатываю {col}...")
            new_cols = [f"{col}_dev", f"{col}_status"]
            df[new_cols] = df.apply(lambda row: calculatemetrics(row, col), axis=1)

    # 4. расчетные индексы
    print("считаю индексы (homa, de_ritis)...")
    df['idx_homa'] = (df['biopro_LBDSGLSI'] * (df['ins_LBDINSI'] / 6.0)) / 22.5
    df['idx_deritis'] = df['biopro_LBXSASSI'] / (df['biopro_LBXSATSI'] + 0.0001)

    # 5. сохранение
    df.to_csv(output_path, index=False)
    print(f"готово! чекай {output_path}")

if __name__ == "__main__":
    runnormalization()