import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

# загружаем данные
df = pd.read_csv('data/processed/datasets/vectorized_dataset.csv')

# 1. отбираем только колонки статусов и таргеты
status_cols = [c for c in df.columns if c.endswith('_status')]
targets = ['target_diabetes', 'target_anemia']
df_arm = df[status_cols + targets].copy()

# 2. маппер для превращения чисел в понятные слова
def statustolabel(val, col_name):
    if pd.isna(val) or val == 0: 
        return None # игнорируем норму и пустые значения, нам интересны только отклонения
    
    prefix = col_name.replace('_status', '') 
    
    mapping = {
        1: 'borderline_high',
        2: 'high',
        3: 'critical_high',
        -1: 'borderline_low',
        -2: 'low',
        -3: 'critical_low'
    }
    
    return f"{prefix}_{mapping.get(val)}"

# 3. применяем маппинг
print("превращаю цифры в слова...")
transactions = []

for idx, row in df_arm.iterrows():
    current_transaction = []
    
    for col in status_cols:
        label = statustolabel(row[col], col)
        if label:
            current_transaction.append(label)
            
    if row['target_diabetes'] == 1:
        current_transaction.append('DIABETES')
    if row['target_anemia'] == 1:
        current_transaction.append('ANEMIA')
        
    if current_transaction:
        transactions.append(current_transaction)

print(f"подготовлено {len(transactions)} корзин для анализа.")

# 4. векторизация для mlxtend
print("строю матрицу признаков...")
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# 5. поиск частых наборов (frequent itemsets)
print("запускаю fp-growth (ищу частые сочетания)...")
frequent_itemsets = fpgrowth(df_onehot, min_support=0.02, use_colnames=True)

# 6. генерация правил (association rules)
print("генерирую правила...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# 7. фильтруем правила, где в результате (consequents) стоит DIABETES или ANEMIA
target_rules = rules[rules['consequents'].apply(lambda x: 'DIABETES' in x or 'ANEMIA' in x)]

# сортируем по уверенности (confidence)
target_rules = target_rules.sort_values(by='confidence', ascending=False)

print("\nтоп правил:")
print(target_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# сохраняем в csv для истории
target_rules.to_csv('data/processed/association_rules.csv', index=False)