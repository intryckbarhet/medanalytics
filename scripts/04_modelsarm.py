import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import re

# пути
data_path = 'data/processed/dataset.csv'
rules_path = 'data/processed/associationrules.csv'
model_dir = 'models'

def parse_rule_item(item):
    """превращает 'glucose_crit' в ('glucose_status', 3.0)"""
    mapping = {'crit': 3, 'high': 2, 'border': 1, 'border_low': -1, 'low': -2, 'crit_low': -3}
    for suffix, val in mapping.items():
        if item.endswith(f'_{suffix}'):
            col_base = item.rsplit(f'_{suffix}', 1)[0]
            return f"{col_base}_status", float(val)
    return None, None

def train_with_arm():
    df = pd.read_csv(data_path)
    rules = pd.read_csv(rules_path)
    
    # 1.    выбираем лучшие правила (lift > 2 и уверенность > 0.5)
    rules = rules[(rules['lift'] > 2) & (rules['confidence'] > 0.5)].head(100)
    
    print(f"внедряю {len(rules)} правил как фичи...")
    
    rule_features = []
    for i, row in rules.iterrows():
        # достаем названия из "frozenset({'item1', 'item2'})"
        items = re.findall(r"'(.*?)'", row['antecedents'])
        
        mask = pd.Series(True, index=df.index)
        is_valid = False
        
        for item in items:
            if 'TARGET_' in item: continue # пропускаем, если в условии другой диагноз
            
            col_name, target_val = parse_rule_item(item)
            if col_name in df.columns:
                mask &= (df[col_name] == target_val)
                is_valid = True
        
        if is_valid:
            col_id = f"arm_rule_{i}"
            df[col_id] = mask.astype(int)
            rule_features.append(col_id)

    # 2.    готовим список всех фич
    base_features = [c for c in df.columns if '_dev' in c or '_status' in c or 'idx_' in c]
    base_features += ['gender', 'age']
    base_features += [c for c in df.columns if c.startswith('feature_')]

    all_features = base_features + rule_features
    
    target_cols = [c for c in df.columns if c.startswith('target_')]
    summary = {}

    print(f"обучаю модели на {len(all_features)} признаках...")

    for target in target_cols:
        df_tmp = df.dropna(subset=[target]).copy()
        if df_tmp[target].sum() < 100: continue

        X = df_tmp[all_features]
        y = df_tmp[target].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = CatBoostClassifier(
            iterations=1000,
            depth=5,
            learning_rate=0.03,
            early_stopping_rounds=50,
            scale_pos_weight=weight,
            eval_metric='AUC',
            random_seed=42,
            verbose=0,
            allow_writing_files=False
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
        
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        summary[target] = round(auc, 4)
        
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        model.save_model(os.path.join(model_dir, f"{target}.cbm"))
        print(f"модель {target} готова. AUC: {auc:.4f}")

    print("\nитоговый AUC с ARM-подсказками:")
    for t, a in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"{t:.<30} {a}")

if __name__ == "__main__":
    train_with_arm()