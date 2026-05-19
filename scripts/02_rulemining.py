import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np

df = pd.read_csv('data/processed/dataset.csv')

# 1.    определяем колонки
status_cols = [c for c in df.columns if c.endswith('_status')]
target_cols = [c for c in df.columns if c.startswith('target_')]

# 2.    маппер
def statustolabel(val, col_name):
    if pd.isna(val) or val == 0: return None
    prefix = col_name.replace('_status', '')
    mapping = {1:'border', 2:'high', 3:'crit', -1:'border_low', -2:'low', -3:'crit_low'}
    return f"{prefix}_{mapping.get(val)}"

# 3.    сбор корзин
print("собираю корзины...")
transactions = []
for _, row in df.iterrows():
    items = [statustolabel(row[c], c) for c in status_cols if statustolabel(row[c], c)]
    # только те таргеты, которые == 1
    items += [t.upper() for t in target_cols if row[t] == 1.0]
    if items: transactions.append(items)

# 4.    векторизация
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# 5.    поиск правил (min_support 1% для редких болячек)
print("ищу правила...")
frequent = fpgrowth(df_onehot, min_support=0.01, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.3)

# 6.    фильтр: только те, где в конце стоит любой таргет
targets_upper = [t.upper() for t in target_cols]
final_rules = rules[rules['consequents'].apply(lambda x: any(item in targets_upper for item in x))]

final_rules = final_rules.sort_values(['confidence', 'lift'], ascending=False)

# 7.    сохранение
final_rules.to_csv('data/processed/associationrules.csv', index=False)
print(f"готово. найдено правил: {len(final_rules)}")
print(final_rules[['antecedents', 'consequents', 'confidence', 'lift']].head(10))