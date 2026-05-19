import pandas as pd

# загрузка
rules = pd.read_csv('data/processed/associationrules.csv')

def audit_rules():
    # 1. ищем всё, что ведет к печени
    liver_rules = rules[rules['consequents'].str.contains('TARGET_LIVER_CONDITION', case=False)]
    
    print("=== ПРАВИЛА, УКАЗЫВАЮЩИЕ НА ПРОБЛЕМЫ С ПЕЧЕНЬЮ ===")
    if not liver_rules.empty:
        # выводим топ по силе связи (Lift)
        print(liver_rules[['antecedents', 'confidence', 'lift']].head(20))
    else:
        print("по печени правил не найдено. скорее всего, связь слишком слабая для порога min_support.")

    # 2. ищем "универсальные улики" (признаки, которые чаще всего встречаются в правилах)
    # это покажет, какие анализы вообще самые информативные в твоем датасете
    all_antecedents = rules['antecedents'].str.extractall(r"'(.*?)'")[0]
    top_features = all_antecedents.value_counts().head(15)
    
    print("\n=== ТОП-15 ПРИЗНАКОВ-СТУКАЧЕЙ (частота в правилах) ===")
    print(top_features)

    # 3. ищем "комбо-убийцы" (правила с самым высоким Lift вообще)
    print("\n=== ТОП-10 САМЫХ СУРОВЫХ ЗАКОНОМЕРНОСТЕЙ (Max Lift) ===")
    print(rules.sort_values(by='lift', ascending=False)[['antecedents', 'consequents', 'lift']].head(10))

if __name__ == "__main__":
    audit_rules()