import numpy as np
from scipy import stats
from generator import generate_business_data

from scipy.stats import norm

np.random.seed(42)
n_users = 2000
iterations = 1000

success = 0
for i in range(iterations):
    conversion_a = 0.1
    _, group_a, _ = generate_business_data(n = n_users,conversion = conversion_a)
    conversion_b = 0.12
    _, group_b, _ = generate_business_data(n = n_users,conversion = conversion_b)
    stat, p_value = stats.ttest_ind(group_a, group_b, equal_var = False)


    if (p_value < 0.05):
        success += 1

power = success/iterations

print(f'Мощность составляет {power:.3f}')

#фиксируем параметры эксперимента
N = 2000              
p1 = 0.10             
alpha = 0.05
beta = 1 - 0.8        

S2 = p1 * (1 - p1)

#MDE
MDE = (norm.ppf(1 - alpha) + norm.ppf(1 - beta)) * np.sqrt(2 * S2 / N)

print(f"MDE при N={N}: {MDE:.4f} (то есть {MDE*100:.2f} п.п.)")

