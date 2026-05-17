import pandas as pd
import os
from functools import reduce

# пути к папкам
raw_dir = 'data/raw'
proc_dir = 'data/processed'

'''
*   **DEMO_L:** `SEQN` (id), `RIAGENDR` (пол), `RIDAGEYR` (возраст).
*   **BMX_L:** `SEQN`, `BMXWT` (вес), `BMXHT` (рост), `BMXBMI` (имт), `BMXWAIST` (талия).
*   **BIOPRO_L:** `SEQN`, `LBXSGL` (глюкоза), `LBXSASSI` (аст), `LBXSATSI` (алт), `LBXSCR` (креатинин), `LBXSTP` (белок), `LBXSCH` (холестерин).
*   **CBC_L:** `SEQN`, `LBXWBCSI` (лейкоциты), `LBXRBCSI` (эритроциты), `LBXHGB` (гемоглобин), `LBXMCVSI` (mcv), `LBXPLTSI` (тромбоциты).
*   **DIQ_L (таргет диабет):** `SEQN`, `DIQ010` (диагноз).
*   **MCQ_L (таргет анемия):** `SEQN`, `MCQ053` (диагноз).
*   **FERTIN_L:** `SEQN`, `LBXFER` (ферритин).
*   **GHB_L:** `SEQN`, `LBXGH` (гликированный гемоглобин).
*   **INS_L:** `SEQN`, `LBDINSI` (инсулин).
*   **HSQ_L:** `SEQN`, `HSD010` (общее самочувствие).
'''

dc = {
    'FERTIN_L.xpt': ['SEQN', 'LBXFER'],
    'BIOPRO_L.xpt': ['SEQN', 'LBXSGL', 'LBXSASSI', 'LBXSATSI', 'LBXSCR', 'LBXSTP', 'LBXSCH'],
    'DEMO_L.xpt': ['SEQN', 'RIAGENDR', 'RIDAGEYR'],
    'BMX_L.xpt': ['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI'],
    'GHB_L.xpt': ['SEQN', 'LBXGH'],
    'DIQ_L.xpt': ['SEQN', 'DIQ010'],
    'CBC_L.xpt': ['SEQN', 'LBXWBCSI', 'LBXRBCSI', 'LBXHGB', 'LBXMCVSI', 'LBXPLTSI'],
    'INS_L.xpt': ['SEQN', 'LBDINSI'],
    'MCQ_L.xpt': ['SEQN', 'MCQ053']
}

def processandmerge():
    processed_dfs = []

    for dataset, keep_cols in dc.items():
        path = os.path.join(raw_dir, dataset)
        if not os.path.exists(path):
            print(f"пропускаю {dataset}, файла нет")
            continue
            
        # читаем xpt
        df = pd.read_sas(path)
        
        # оставляем только нужные колонки
        df = df[keep_cols]
        
        # переименовываем всё кроме SEQN
        prefix = dataset.split('_')[0].lower() # берем начало имени файла как префикс
        df = df.rename(columns={col: f'{prefix}_{col}' for col in df.columns if col != 'SEQN'})
        
        # сохраняем промежуточный csv (теперь с f-строкой!)
        out_name = f'{prefix}_cleaned.csv'
        df.to_csv(os.path.join(proc_dir, out_name), index=False)
        print(f"файл {dataset} обработан и сохранен как {out_name}")
        
        processed_dfs.append(df)

    # великое объединение (мерджим всё по очереди по SEQN)
    # начинаем с первого датафрейма и погнали
    if processed_dfs:
        print("начинаю мердж всех файлов...")
        # reduce прогонит merge по всему списку
        final_df = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='left'), processed_dfs)
        
        final_path = os.path.join(proc_dir, 'final_dataset.csv')
        final_df.to_csv(final_path, index=False)
        print(f"готово! итоговая матрица лежит в {final_path}")
        print(f"размер: {final_df.shape}")
    else:
        print("нечего мерджить, список пуст!")

if __name__ == "__main__":
    processandmerge()