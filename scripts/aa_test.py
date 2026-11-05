import numpy as np
import pandas as pd
from scipy import stats
from generator import generate_business_data


np.random.seed(42)
n_users = 2000
iterations = 1000


def aa_run_experiment(n_users):
    common_conv = 0.1
    common_aov = 6000
    _, buyer_a, _ = generate_business_data(n_users, 
                                           conversion = common_conv, 
                                           aov_mean = common_aov)
    _, buyer_b, _ = generate_business_data(n_users, 
                                           conversion = common_conv, 
                                           aov_mean = common_aov)
    stat, p_value = stats.ttest_ind(buyer_a, buyer_b, equal_var = False)
    return p_value



false_positives = 0
for _ in range(iterations):
    p_val = aa_run_experiment(n_users)
    if (p_val < 0.05):
        false_positives += 1

fpr = false_positives / iterations 
print(f'Доля ложноположительных срабатываний в АА-тесте {fpr:.3f}')