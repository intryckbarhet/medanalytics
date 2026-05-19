import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import json

data_path = 'data/processed/dataset.csv'
model_dir = 'models'
if not os.path.exists(model_dir): os.makedirs(model_dir)

def trainmodels():
    df = pd.read_csv(data_path)
    
    # 1.    готовим список фич
    features = [c for c in df.columns if '_dev' in c or '_status' in c or 'idx_' in c]
    features += ['gender', 'age', 'bmi']
    features += [c for c in df.columns if c.startswith('feature_')]
    
    # 2.    ищем все таргеты
    target_cols = [c for c in df.columns if c.startswith('target_')]
    
    summary_results = {}

    print(f"начинаю обучение {len(target_cols)} моделей...")

    for target in target_cols:
        df_tmp = df.dropna(subset=[target]).copy()
        
        # проверка на вшивость
        pos_cases = df_tmp[target].sum()
        if pos_cases < 100:
            print(f"    скипаю {target}: слишком мало данных ({int(pos_cases)} больных)")
            continue

        X = df_tmp[features]
        y = df_tmp[target].astype(int)

        # делим 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # считаем вес для балансировки
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
        summary_results[target] = round(auc, 4)
        
        model.save_model(os.path.join(model_dir, f"{target}.cbm"))
        print(f"модель {target} готова. AUC: {auc:.4f}")

    print("\nитоговая точность (AUC):")
    sorted_res = sorted(summary_results.items(), key=lambda x: x[1], reverse=True)
    for t, a in sorted_res:
        print(f"{t:.<30} {a}")

if __name__ == "__main__":
    trainmodels()