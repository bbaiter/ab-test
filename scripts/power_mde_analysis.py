import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#параметры эксперимента
p1 = 0.10
alpha = 0.05
beta = 1 - 0.8 # 0.2
S2 = p1 * (1 - p1)

n_values = np.arange(500, 5000, 100)

powers = []
mdes = []

for n in n_values:
    #MDE

    #вывели на лекции
    mde = (norm.ppf(1 - alpha/2) + norm.ppf(1 - beta)) * np.sqrt(2 * S2 / n)
    mdes.append(mde * 100) #п.п.

    p2 = 0.12
    z_effect = (p2 - p1) / np.sqrt(2 * S2 / n)
    power = norm.cdf(z_effect - norm.ppf(1 - alpha/2))#площадь правого пересечения квантиля и статистики
    powers.append(power)


# --- Расчет точного N ---
p2 = 0.12
target_mde = 0.02
Z_alpha = norm.ppf(1 - alpha/2)
Z_beta = norm.ppf(1 - beta)
N_exact = int(np.ceil(((Z_alpha + Z_beta)**2 * 2 * S2) / (target_mde**2)))
print(f"Точное N: {N_exact}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# График 1
ax1.plot(n_values, powers, color='blue')
ax1.axhline(y=0.8, color='red', linestyle='--') # Красная линия 80%
ax1.axvline(x=N_exact, color='black', linestyle=':', label=f'N={N_exact}')
ax1.set_xlabel('Размер выборки)')
ax1.set_ylabel('Мощность')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax1.legend()

# График 2
ax2.plot(n_values, mdes, color='green')
ax2.axhline(y=2.0, color='red', linestyle='--') # Красная линия 2 п.п.
ax2.axvline(x=N_exact, color='black', linestyle=':', label=f'N={N_exact}')
ax2.set_xlabel('Размер выборки (N)')
ax2.set_ylabel('MDE (п.п.)')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax2.legend()

plt.tight_layout()
plt.savefig('../plots/design.png')
print("Сохранено в plots/design.png")