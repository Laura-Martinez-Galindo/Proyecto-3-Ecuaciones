import numpy as np
import matplotlib.pyplot as plt

def u(x, t, terms=1000):
    result = 0
    for n in range(1, terms+1):
        cn = 120 / ((2*n - 1)**2 * np.pi**2 * (2 * np.cos(n*np.pi) + (2*n - 1)*np.pi))
        result += cn * np.exp(-(2*n - 1)**2 * np.pi**2 * t / 3600) * np.sin((2*n - 1)*np.pi*x / 60)
    return result

def find_warmest_point(t_values, x_values):
    xm_values = []
    for t in t_values:
        u_values = [u(x, t) for x in x_values]
        max_index = np.argmax(u_values)
        xm_values.append(x_values[max_index])
    return xm_values

# Valores de t
t_values = np.linspace(0, 201, 100)

# Valores de x
x_values = np.linspace(0, 300, 100)

# Encontrar la ubicación del punto más cálido para cada valor de t
xm_values = find_warmest_point(t_values, x_values)

# Índice del punto que queremos etiquetar
index_to_label = 50

# Graficar xm versus t
plt.figure(figsize=(10, 6))
plt.plot(t_values, xm_values)
plt.annotate(f'x_m={xm_values[index_to_label]:.2f}', (t_values[index_to_label], xm_values[index_to_label]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('t')
plt.ylabel('Ubicación del punto más cálido (xm)')
plt.title('Ubicación del punto más cálido en la barra en función del tiempo')
plt.grid(False)
plt.show()

