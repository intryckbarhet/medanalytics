import pandas as pd
import numpy as np
import json
import os
from functools import reduce

# конфиг
config_path = 'data/references/config.json'
raw_dir = 'data/raw'
output_path = 'data/processed/dataset.csv'

with open(config_path, 'r') as f:
    cfg = json.load(f)

def runpipeline():
    processed_dfs = []
    glob = cfg['global_settings']

    # 1.    первичная очистка
    for cat_name, files in cfg['categories'].items():
        for file_name, cols in files.items():
            path = os.path.join(raw_dir, file_name)
            if not os.path.exists(path): continue
            
            df = pd.read_sas(path)[['SEQN'] + list(cols.keys())]

            for raw_col, settings in cols.items():
                new_name = settings['name']
                
                # логика для лаборатории и демографии
                if cat_name in ['labs', 'demographics']:
                    df[raw_col] = df[raw_col].apply(lambda x: x if x > glob['min_valid_numeric'] else np.nan)
                    if 'multiplier' in settings:
                        df[raw_col] *= settings['multiplier']
                    
                    # статы
                    settings['stats'] = {
                        'min': float(df[raw_col].min()),
                        'max': float(df[raw_col].max()),
                        'mean': float(df[raw_col].mean()),
                        'nulls': int(df[raw_col].isnull().sum())
                    }
                
                # логика для опросников
                else:
                    df[raw_col] = df[raw_col].replace(glob['questionnaire_junk'], np.nan)
                    mapping = settings['custom_mapping'] if settings['custom_mapping'] else glob['default_mapping']
                    df[raw_col] = df[raw_col].astype(str).map({str(k): v for k, v in mapping.items()})
                    
                    # статы (value_counts)
                    settings['stats'] = df[raw_col].value_counts().to_dict()

                df = df.rename(columns={raw_col: new_name})
            
            processed_dfs.append(df)

    # 2.    объединение
    final_df = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='left'), processed_dfs)

    # 3.    векторная нормализация
    for cat_name, files in cfg['categories'].items():
        if cat_name != 'labs': continue
        for file_name, cols in files.items():
            for raw_col, settings in cols.items():
                name = settings['name']
                if 'ranges' not in settings: continue
                
                # вычисляем нормы векторно
                if settings['sex_dependent']:
                    # маска для м и ж
                    is_m = final_df['gender'] == 1.0
                    is_f = final_df['gender'] == 2.0
                    m_min, m_max = settings['ranges']['male']['min'], settings['ranges']['male']['max']
                    f_min, f_max = settings['ranges']['female']['min'], settings['ranges']['female']['max']
                    
                    cur_min = np.where(is_m, m_min, np.where(is_f, f_min, np.nan))
                    cur_max = np.where(is_m, m_max, np.where(is_f, f_max, np.nan))
                else:
                    cur_min = settings['ranges']['any']['min']
                    cur_max = settings['ranges']['any']['max']

                width = cur_max - cur_min
                center = (cur_max + cur_min) / 2
                
                # вектор 1:  deviation
                final_df[f'{name}_dev'] = (final_df[name] - center) / width
                
                # вектор 2: status
                dev = final_df[f'{name}_dev']
                abs_dev = dev.abs()
                sign = np.sign(dev)
                
                conditions = [
                    (abs_dev <= 0.35),
                    (abs_dev <= 0.5),
                    (abs_dev <= 1.0)
                ]
                choices = [0, 1 * sign, 2 * sign]
                final_df[f'{name}_status'] = np.select(conditions, choices, default=3 * sign)
                
                final_df.loc[final_df[name].isna(), f'{name}_status'] = np.nan

    final_df['idx_homa'] = (final_df['glucose_random'] * (final_df['insulin'] / 6.0)) / 22.5
    final_df['idx_deritis'] = final_df['ast'] / (final_df['alt'] + 0.0001)

    final_df.to_csv(output_path, index=False)
    
    # обновляем конфиг статами
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    
    print(f"done. shape: {final_df.shape}.")

if __name__ == "__main__":
    runpipeline()