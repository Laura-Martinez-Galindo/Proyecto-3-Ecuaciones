import numpy as np
import matplotlib.pyplot as plt

def u(x, t, terms=1000):
    result = 0
    for n in range(1, terms+1):
        cn = 120 / ((2*n - 1)**2 * np.pi**2 * (2 * np.cos(n*np.pi) + (2*n - 1)*np.pi))
        result += cn * np.exp(-(2*n - 1)**2 * np.pi**2 * t / 3600) * np.sin((2*n - 1)*np.pi*x / 60)
    return result

# Valores de x
x_values = np.linspace(0, 300, 100)

# Valores de t
t_values = np.arange(0, 201, 10)  # Cambiados a valores de 0 a 150 en pasos de 5

plt.figure(figsize=(10, 6))
# Gráficos para cada valor de t
for t in t_values:
    u_values = [u(x, t) for x in x_values]
    plt.plot(x_values, u_values, label=f't={t}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Gráfico de u(x, t) para varios valores de t')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(False)
plt.show()
