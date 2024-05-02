import numpy as np
import matplotlib.pyplot as plt

def f1(t, x1, x2):
    return 9*x1 + 24*x2 + 5*np.cos(t) - (1/3)*np.sin(t)

def f2(t, x1, x2):
    return -24*x1 - 51*x2 - 9*np.cos(t) + (1/3)*np.sin(t)

def runge_kutta(x1_0, x2_0, t_end, h):
    n_steps = int(t_end / h)
    t_values = np.linspace(0, t_end, n_steps + 1)
    x1_values = np.zeros(n_steps + 1)
    x2_values = np.zeros(n_steps + 1)

    x1_values[0] = x1_0
    x2_values[0] = x2_0

    for i in range(n_steps):
        t_n = t_values[i]
        x1_n = x1_values[i]
        x2_n = x2_values[i]

        k1_x1 = h * f1(t_n, x1_n, x2_n)
        k1_x2 = h * f2(t_n, x1_n, x2_n)

        k2_x1 = h * f1(t_n + 0.5*h, x1_n + 0.5*k1_x1, x2_n + 0.5*k1_x2)
        k2_x2 = h * f2(t_n + 0.5*h, x1_n + 0.5*k1_x1, x2_n + 0.5*k1_x2)

        k3_x1 = h * f1(t_n + 0.5*h, x1_n + 0.5*k2_x1, x2_n + 0.5*k2_x2)
        k3_x2 = h * f2(t_n + 0.5*h, x1_n + 0.5*k2_x1, x2_n + 0.5*k2_x2)

        k4_x1 = h * f1(t_n + h, x1_n + k3_x1, x2_n + k3_x2)
        k4_x2 = h * f2(t_n + h, x1_n + k3_x1, x2_n + k3_x2)

        x1_values[i+1] = x1_n + (1/6)*(k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1)
        x2_values[i+1] = x2_n + (1/6)*(k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2)

    return t_values, x1_values, x2_values

# Valores iniciales
x1_0 = 4/3
x2_0 = 2/3
t_end = 20

# Valores de h
h_values = [0.2, 0.1] + [2**(-k)/15 for k in range(3)]

# Solución exacta
def exact_solution(t):
    x1_exact = 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)
    x2_exact = -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)
    return x2_exact

# Calcular errores para cada valor de h
errors = []
for h in h_values:
    t_values, x1_values, x2_values = runge_kutta(x1_0, x2_0, t_end, h)
    x2_exact_values = exact_solution(t_values)
    error = np.abs(x2_values - x2_exact_values)
    errors.append(error[-1])

# Graficar errores en escala logarítmica
plt.figure(figsize=(8, 6))
plt.semilogx(h_values, errors, marker='o', label='Error absoluto')
plt.xlabel('Valor de h')
plt.ylabel('Error absoluto en x2(20)')
plt.title('Error absoluto en la aproximación de x2(20)')
plt.grid(True)
plt.legend()
plt.show()