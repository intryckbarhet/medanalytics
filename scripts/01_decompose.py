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
'''

dc = {
    'DEMO_L.xpt': ['SEQN', 'RIAGENDR', 'RIDAGEYR'],
    
    'BMX_L.xpt': ['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI'],
    'FERTIN_L.xpt': ['SEQN', 'LBXFER'],
    'BIOPRO_L.xpt': ['SEQN', 'LBDSGLSI', 'LBXSASSI', 'LBXSATSI', 'LBDSCRSI', 'LBDSTPSI', 'LBDSCHSI', 'LBDSIRSI'],
    'GHB_L.xpt': ['SEQN', 'LBXGH'],
    'CBC_L.xpt': ['SEQN', 'LBXWBCSI', 'LBXRBCSI', 'LBXHGB', 'LBXMCVSI', 'LBXPLTSI'],
    'INS_L.xpt': ['SEQN', 'LBDINSI'],

    # таргеты диабет
    'DIQ_L.xpt': ['SEQN', 'DIQ010',    # диабет (1=да, 2=нет, 3=погран)
                  'DIQ160',            # преддиабет (1=да, 2=нет)
                  'DIQ050'],           # на инсулине (1=да, 2=нет)
    
    # таргеты из MCQ
    'MCQ_L.xpt': ['SEQN', 'MCQ053']    # анемия (1=да, 2=нет)

}

# мусорные значения в опросниках
JUNK_VALUES = [7, 7.0, 9, 9.0, 777, 777.0, 999, 999.0]
# колонки которые надо чистить от мусора
QUESTIONNAIRE_COLS = [
    'diq_DIQ010', 'diq_DIQ160', 'diq_DIQ050', 'mcq_MCQ053'
]
HGB_COL = 'cbc_LBXHGB'

def clean(df):
    # чистим мусорные ответы в опросниках
    for col in QUESTIONNAIRE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(JUNK_VALUES, pd.NA)

    # конвертируем гемоглобин в г/л
    if HGB_COL in df.columns:
        df[HGB_COL] = df[HGB_COL] * 10

    # перекодируем таргеты в 0/1
    # диабет: 1=да, 3=пограничный - 1, 2=нет - 0
    if 'diq_DIQ010' in df.columns:
        df['target_diabetes'] = df['diq_DIQ010'].map(
            {1.0: 1, 3.0: 1, 2.0: 0}
        )

    # преддиабет: 1=да - 1, 2=нет - 0
    if 'diq_DIQ160' in df.columns:
        df['target_prediabetes'] = df['diq_DIQ160'].map(
            {1.0: 1, 2.0: 0}
        )

    # на инсулине: 1=да - 1, 2=нет - 0
    if 'diq_DIQ050' in df.columns:
        df['target_insulin'] = df['diq_DIQ050'].map(
            {1.0: 1, 2.0: 0}
        )

    # анемия: 1=да - 1, 2=нет - 0
    if 'mcq_MCQ053' in df.columns:
        df['target_anemia'] = df['mcq_MCQ053'].map(
            {1.0: 1, 2.0: 0}
        )

    # выбросы: ферритин не может быть почти нулём
    if 'fertin_LBXFER' in df.columns:
        df.loc[df['fertin_LBXFER'] < 1, 'fertin_LBXFER'] = pd.NA
    
    # выбросы: возраст не может быть почти нулём
    if 'demo_RIDAGEYR' in df.columns:
        df.loc[df['demo_RIDAGEYR'] < 1, 'demo_RIDAGEYR'] = pd.NA

    print("после очистки:")
    print(f"  target_diabetes:    {df['target_diabetes'].value_counts().to_dict()}")
    print(f"  target_anemia:      {df['target_anemia'].value_counts().to_dict()}")

    return df


def processandmerge():
    processed_dfs = []

    for dataset, keep_cols in dc.items():
        path = os.path.join(raw_dir, dataset)
        if not os.path.exists(path):
            print(f"пропускаю {dataset}, файла нет")
            continue

        df = pd.read_sas(path)
        df = df[keep_cols]

        prefix = dataset.split('_')[0].lower()
        df = df.rename(columns={
            col: f'{prefix}_{col}' for col in df.columns if col != 'SEQN'
        })

        out_name = f'{prefix}_cleaned.csv'
        df.to_csv(os.path.join(proc_dir, out_name), index=False)
        print(f"файл {dataset} обработан → {out_name}")

        processed_dfs.append(df)

    if processed_dfs:
        print("начинаю мердж...")
        final_df = reduce(
            lambda left, right: pd.merge(left, right, on='SEQN', how='left'),
            processed_dfs
        )

        # очистка перед сохранением
        final_df = clean(final_df)

        final_path = os.path.join(proc_dir, 'final_dataset.csv')
        final_df.to_csv(final_path, index=False)
        print(f"готово! итоговая матрица → {final_path}")
        print(f"размер: {final_df.shape}")
    else:
        print("нечего мерджить!")


if __name__ == "__main__":
    processandmerge()
