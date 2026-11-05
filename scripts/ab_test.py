import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

df = pd.read_csv('../data/gen_data.csv')

counts = df['group'].value_counts()
n_a, n_b = counts['group_a'], counts['group_b']
_, p_srm = stats.chisquare([n_a, n_b])
_, p_plat, _, _ = stats.chi2_contingency(pd.crosstab(df['group'], df['platform']))
print(f"SRM размер: {p_srm:.3f} | SRM платформы: {p_plat:.3f}")

a_conv = df[df['group'] == 'group_a']['if_buyer']
b_conv = df[df['group'] == 'group_b']['if_buyer']
count = [b_conv.sum(), a_conv.sum()]
nobs = [len(b_conv), len(a_conv)]
stat_z, p_conv = proportions_ztest(count, nobs)

#интервал
cm_conv = CompareMeans(DescrStatsW(b_conv), DescrStatsW(a_conv))
ci_low_conv, ci_high_conv = cm_conv.tconfint_diff(alpha=0.05, usevar='unequal')
lift_conv = b_conv.mean() - a_conv.mean()

print(f"Конверсия")
print(f"конверсия группы A: {a_conv.mean()*100:.2f}%, конверсия группы B: {b_conv.mean()*100:.2f}%")
print(f"разница между группой B и А: {lift_conv*100:.2f} п.п.")
print(f"95% доверительный интервал: [{ci_low_conv*100:.2f}, {ci_high_conv*100:.2f}] п.п.")
print(f"p-value: {p_conv:.4f}\n")

a_arpu = df[df['group'] == 'group_a']['revenue']
b_arpu = df[df['group'] == 'group_b']['revenue']

stat_t, p_arpu = stats.ttest_ind(b_arpu, a_arpu, equal_var=False)
cm_arpu = CompareMeans(DescrStatsW(b_arpu), DescrStatsW(a_arpu))
ci_low_arpu, ci_high_arpu = cm_arpu.tconfint_diff(alpha=0.05, usevar='unequal')
lift_arpu = b_arpu.mean() - a_arpu.mean()

print(f"средняя выручка с пользователя(ARPU)")
print(f"ARPU A: {a_arpu.mean():.2f} руб., ARPU B: {b_arpu.mean():.2f} руб.")
print(f"разница между группой B и A: {lift_arpu:.2f} руб.")
print(f"95% доверительный интервал: [{ci_low_arpu:.2f}, {ci_high_arpu:.2f}] руб.")
print(f"p-value: {p_arpu:.4f}")