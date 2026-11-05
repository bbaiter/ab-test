import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans

def ab_test(dataset):
    a = dataset[dataset['group'] == 'group_a']['if_buyer']
    b = dataset[dataset['group'] == 'group_b']['if_buyer']
    #equal_var = False -- предполагаем, что дисперсии разные
    #ищем p-value
    stat, p_value = stats.ttest_ind(a, b, equal_var = False)
    cm = CompareMeans(DescrStatsW(b), DescrStatsW(a))
    #сам интервал
    #
    left, right = cm.tconfint_diff(alpha = 0.05, usevar='unequal')
    print(f'Доверительный интервал ({left:.4f},{right:.4f})')

    print(f'p_value = {p_value:.4f}' )

np.random.seed(42)

group_size = 3230

conv_a = 0.1
conv_b = 0.12

group_a = np.random.binomial(1, conv_a, group_size)
group_b = np.random.binomial(1, conv_b, group_size)

#склеим -> в один dataframe
df = pd.DataFrame({
    'user_id' : range(0, group_size*2),
    'group' : ['group_a']*group_size + ['group_b']*group_size,
    'if_buyer' : np.concatenate([group_a, group_b])
})
ab_test(dataset=df)
#print(df)
#print(df.groupby('group')['if_buyer'].mean())
