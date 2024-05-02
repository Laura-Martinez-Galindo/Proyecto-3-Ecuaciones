from math import cos, sin, exp, log
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def f(t, x1, x2):
    return 9 * x1 + 24 * x2 + 5 * cos(t) - 1/3 * sin(t)

def g(t, x1, x2):
    return -24 * x1 - 51 * x2 - 9 * cos(t) + 1/3 * sin(t)

def solucion_real(t):
    x1 = 2*exp(-3*t) -exp(-39*t) + 1/3*cos(t)
    x2 = -exp(-3*t) + 2*exp(-39*t) - 1/3*cos(t)
    
    return x1, x2

errores = {'h': [], 'x2_aproximado': [], 'x2_real': [], 'error_x2': []}

def runge_kutta(t_0, y_0, t_f, h):
    
    n = int((t_f - t_0) / h)
    datos = {'n': [], 't_n': [], 'x1_n': [], 'x2_n': [], 'k1n_x1': [], 'k1n_x2': [], 
            'k2n_x1': [], 'k2n_x2': [], 'k3n_x1': [], 'k3n_x2': [], 'k4n_x1': [], 'k4n_x2': []}
    
    x1_0, x2_0 = y_0
    datos['t_n'].append(t_0)
    datos['x1_n'].append(x1_0)
    datos['x2_n'].append(x2_0)

    for i in range(n):
        #Iteración n
        t_n = datos["t_n"][-1]
        x1_n = datos["x1_n"][-1]
        x2_n = datos["x2_n"][-1]
        
        k1n_x1 = f(t_n, x1_n, x2_n)
        k1n_x2 = g(t_n, x1_n, x2_n)
        
        k2n_x1 = f(t_n + h/2, x1_n + (1/2)*h*k1n_x1, x2_n + (1/2)*h*k1n_x2)
        k2n_x2 = g(t_n + h/2, x1_n + (1/2)*h*k1n_x1, x2_n + (1/2)*h*k1n_x2)
        
        k3n_x1 = f(t_n + h/2, x1_n + (1/2)*h*k2n_x1, x2_n + (1/2)*h*k2n_x2)
        k3n_x2 = g(t_n + h/2, x1_n + (1/2)*h*k2n_x1, x2_n + (1/2)*h*k2n_x2)
        
        k4n_x1 = f(t_n + h, x1_n + h*k3n_x1, x2_n + h*k3n_x2)
        k4n_x2 = g(t_n + h, x1_n + h*k3n_x1, x2_n + h*k3n_x2)
        
        
        #Iteración n+1 (actual)
        t_n1 = t_0 + h*i
        x1_n1 = x1_n + (h/6)*(k1n_x1 + 2*k2n_x1 + 2*k3n_x1 + k4n_x1)
        x2_n1 = x2_n + (h/6)*(k1n_x2 + 2*k2n_x2 + 2*k3n_x2 + k4n_x2)
        
        datos['n'].append(i)
        datos['t_n'].append(t_n1)
        datos['x1_n'].append(x1_n1)
        datos['x2_n'].append(x2_n1)

        datos['k1n_x1'].append(k1n_x1)
        datos['k1n_x2'].append(k1n_x2)
        datos['k2n_x1'].append(k2n_x1)
        datos['k2n_x2'].append(k2n_x2)
        datos['k3n_x1'].append(k3n_x1)
        datos['k3n_x2'].append(k3n_x2)
        datos['k4n_x1'].append(k4n_x1)
        datos['k4n_x2'].append(k4n_x2)
        
    x1_real_20, x2_real_20 = solucion_real(t_f)
    x2_aprox_20 = datos['x2_n'][-1]
    error = float(abs(x2_aprox_20 - x2_real_20))
    
    errores.append((h, x2_aprox_20, x2_real_20, error))
    
    errores['h'].append(h)
    errores['x2_aproximado'].append(x2_aprox_20)
    errores['x2_real'].append(x2_real_20)
    errores['error_x2'].append(error)
    
    return datos

# Entradas
t_0 = 0
y_0 = [4/3, 2/3]
t_f = 20
valores_h = [0.2, 0.1] + [2**(-k)/15 for k in range(5)]

#Solución real
t_real = np.linspace(t_0, t_f, 1000)
x1_real = []
x2_real = []
for t in t_real:
    x1, x2 = solucion_real(t)
    x1_real.append(x1)
    x2_real.append(x2)

#Soluciones aproximadas para cada h
archivo_excel = pd.ExcelWriter('runge_kutta.xlsx', engine='xlsxwriter')
for h in valores_h:
    #Iteraciones
    datos = runge_kutta(t_0, y_0, t_f, h)
    tabla_datos = pd.DataFrame(datos)
    tabla_datos.to_excel(archivo_excel, sheet_name=f'h_{h:.4f}', index=False)
    #Gráfica iteraciones, x1 vs t, x2 vs t
    plt.figure(figsize=(10, 6))
    plt.plot(tabla_datos['t_n'], tabla_datos['x1_n'], label=f'x1 (h={h:.4f})', color='blue')
    plt.plot(tabla_datos['t_n'], tabla_datos['x2_n'], label=f'x2 (h={h:.4f})', color='blue')
    plt.scatter(tabla_datos['t_n'], tabla_datos['x1_n'], color='blue')
    plt.scatter(tabla_datos['t_n'], tabla_datos['x2_n'], color='blue')
    plt.fill_between(tabla_datos['t_n'], tabla_datos['x1_n'] - 0.01, tabla_datos['x1_n']+ 0.01, color='blue', alpha=0.2)  # Agregar sombra
    plt.fill_between(tabla_datos['t_n'], tabla_datos['x2_n'] - 0.01, tabla_datos['x2_n']+ 0.01, color='blue', alpha=0.2)  # Agregar sombra
    plt.plot(t_real, x1_real, label='x1 (real)', color="orange")
    plt.plot(t_real, x2_real, label='x2 (real)', color="orange")
    #Gráfica iteraciones, x1 vs x2
    plt.figure(figsize=(10, 6))
    plt.plot(tabla_datos['x1_n'], tabla_datos['x2_n'], label='Solución Aproximada', color="blue")
    plt.scatter(tabla_datos['x1_n'], tabla_datos['x2_n'], color="blue")
    plt.fill_between(tabla_datos['x1_n'], tabla_datos['x2_n'] - 0.01, tabla_datos['x2_n']+ 0.01, color='blue', alpha=0.2)  # Agregar sombra
    plt.plot(x1_real, x2_real, label='Solución Real', color="orange")

    
# Tabla de errores
errores_x2 = errores["error_x2"]
errores["ln(error_x2)"] = []
for error_x2 in errores_x2:
    errores["ln(error_x2)"].append(log(error_x2))

tabla_errores = pd.DataFrame(errores)
tabla_datos.to_excel(archivo_excel, sheet_name=f'errores', index=False)

# Graficr h vs ln(error)
plt.figure(figsize=(10, 6))
plt.plot(tabla_errores['h'], tabla_errores['ln_error'], color="blue")
plt.scatter(tabla_errores['h'], tabla_errores['ln_error'], color="blue")
plt.xlabel('h')
plt.ylabel('ln(error)')
plt.title('h vs ln(error)')
plt.grid(False)
plt.show()
