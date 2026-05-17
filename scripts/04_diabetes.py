import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

# 1. загрузка
print("загружаю данные...")
df = pd.read_csv('data/processed/datasets/vectorized_dataset.csv')

# 2. отбор признаков (feature selection)
# беру только:
# - векторы отклонения (_dev)
# - медицинские статусы (_status)
# - умные индексы (idx_)
# - возраст и пол (база)
features = [c for c in df.columns if '_dev' in c or '_status' in c or 'idx_' in c]
features += ['demo_RIAGENDR', 'demo_RIDAGEYR']

# 3. подготовка таргета
# выкидываем тех, у кого нет метки диабета (на чем нам учиться-то?)
df_model = df.dropna(subset=['target_diabetes']).copy()

X = df_model[features]
y = df_model['target_diabetes'].astype(int)

print(f"финальный размер выборки для обучения: {X.shape}")
print(f"баланс классов: {y.value_counts(normalize=True).to_dict()}")

# 4. делим на train и test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("данные готовы к бою")
# 5. считаем вес для балансировки классов
# (количество здоровых / количество больных)
zero_count = (y_train == 0).sum()
one_count = (y_train == 1).sum()
balance_weight = zero_count / one_count

print(f"вес для балансировки: {balance_weight:.2f}")

# 6. инициализируем модель
model = CatBoostClassifier(
    iterations=1000,         
    depth=5,                 
    learning_rate=0.03,      
    loss_function='Logloss', 
    eval_metric='AUC',       
    scale_pos_weight=balance_weight, 
    random_seed=42,
    verbose=100             
)

# 7. запускаем обучение
print("начинаю обучение...")
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test), 
    early_stopping_rounds=50,  
    use_best_model=True
)

# 8. сохраняем результат
if not os.path.exists('models'): os.makedirs('models')
model.save_model('models/diabetes_model.cbm')
print("модель сохранена в models/diabetes_model.cbm")

# 9. выводим важность фич 
importance = model.get_feature_importance(type='FeatureImportance')
feat_importances = pd.Series(importance, index=X.columns).sort_values(ascending=False)

print("\nглавный виновник диабета:")
print(feat_importances.head(10))

# 10. SHAP 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

