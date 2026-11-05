import numpy as np
import pandas as pd

np.random.seed(42)
n_users = 3532

def generate_business_data(n, conversion, aov_mean= 5000):
    web_trafic = 0.5
    ios_trafic = 0.3
    android_trafic = 0.2
    platforms = np.random.choice(['Web', 'iOS', 'Android'], size=n, p=[web_trafic, ios_trafic, android_trafic])
    #массив покупателей
    is_buyer = np.random.binomial(1, conversion, n)
    #генерим чеки из нормального распределения с заданным средним и выбранным разбросом
    std = aov_mean * 0.3
    order_value = np.where(
        is_buyer == 1, 
        np.random.normal(aov_mean, std, n), # разброс 30% от среднего
        0
    )
    order_value = np.maximum(0, np.round(order_value, 0))
    
    return platforms, is_buyer, order_value

conversion_a = 0.1
aov_mean_a = 6000
plat_a, buyer_a, revenue_a = generate_business_data(n_users,
                                                    conversion=conversion_a, 
                                                    aov_mean=aov_mean_a)

df_a = pd.DataFrame({
    'user_id': range(0, n_users),
    'group': 'group_a',
    'platform': plat_a,
    'if_buyer': buyer_a,
    'revenue': revenue_a
})

conversion_b = 0.12
aov_mean_b = 5500# - пусть возросла конверсия и уменьшилась метрика среднего чека
plat_b, buyer_b, revenue_b = generate_business_data(n_users, 
                                                    conversion=conversion_b,
                                                    aov_mean=aov_mean_b)

df_b = pd.DataFrame({
    'user_id': range(n_users, n_users * 2),
    'group': 'group_b',
    'platform': plat_b,
    'if_buyer': buyer_b,
    'revenue': revenue_b
})

df = pd.concat([df_a, df_b], ignore_index=True)

#print(df.head(10))
#df.to_csv('../data/gen_data.csv', index=False)