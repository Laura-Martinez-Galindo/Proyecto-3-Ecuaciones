from math import cos, sin, exp, log
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def solucion_real(t):
    x1 = 2*exp(-3*t) -exp(-39*t) + 1/3*cos(t)
    x2 = -exp(-3*t) + 2*exp(-39*t) - 1/3*cos(t)
    
    return x1, x2

def sistema_implicito(x1_n, x2_n, t_n1, h):
    
    x1_n1, x2_n1 = sp.symbols('x1_n1 x2_n1')
    
    ecuacion_1 = x1_n1 - (x1_n + h * (9*x1_n1 + 24*x2_n1 + 5*sp.cos(t_n1) - (1/3)*sp.sin(t_n1)))
    ecuacion_2 = x2_n1 - (x2_n + h * (-24*x1_n1 - 51*x2_n1 - 9*sp.cos(t_n1) + (1/3)*sp.sin(t_n1)))

    solucion = sp.solve((ecuacion_1, ecuacion_2), (x1_n1, x2_n1))
    
    return solucion[x1_n1], solucion[x2_n1]

errores = {'h': [], 'x2_aproximado': [], 'x2_real': [], 'error_x2': []}

def euler_implicito(t_0, y_0, t_f, h):
    
    n = int((t_f - t_0) / h)
    datos = {'n': [], 't_n': [], 'x1_n': [], 'x2_n': []}
    
    #Iteración 0
    x1_0, x2_0 = y_0
    
    datos['n'].append(0)
    datos['t_n'].append(t_0)
    datos['x1_n'].append(x1_0)
    datos['x2_n'].append(x2_0)
    
    #Iteración 1 en adelante
    for i in range(1, n+1):
        
        #Iteración n
        x1_n = datos["x1_n"][-1]
        x2_n = datos["x2_n"][-1]
        
        #Iteración n+1 (actual)
        t_n1 = t_0 + h*i
        x1_n1, x2_n1 = sistema_implicito(x1_n, x2_n, t_n1, h)
            
        datos['n'].append(i)
        datos['t_n'].append(t_n1)
        datos['x1_n'].append(x1_n1)
        datos['x2_n'].append(x2_n1)
        
    x1_real_20, x2_real_20 = solucion_real(t_f)
    x2_aprox_20 = datos['x2_n'][-1]
    error = float(abs(x2_aprox_20 - x2_real_20))
    
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
    datos = euler_implicito(t_0, y_0, t_f, h)
    tabla_datos = pd.DataFrame(datos)
    #tabla_datos.to_excel(archivo_excel, sheet_name=f'h_{h:.4f}', index=False)
    #Gráfica iteraciones, x1 vs t, x2 vs t
    plt.figure(figsize=(10, 6))
    plt.title(f"Solución Aproximada vs Solución Real (h={h:.4f})")
    plt.plot(tabla_datos['t_n'], tabla_datos['x1_n'], label=f'x1 aprox', color='g')
    plt.plot(tabla_datos['t_n'], tabla_datos['x2_n'], label=f'x2 aprox', color='g')
    plt.scatter(tabla_datos['t_n'], tabla_datos['x1_n'], color='g')
    plt.scatter(tabla_datos['t_n'], tabla_datos['x2_n'], color='g') 
    plt.plot(t_real, x1_real, label='x1 real', color="orange")
    plt.plot(t_real, x2_real, label='x2 real', color="orange")
    plt.xlabel('t')
    plt.ylabel('x1, x2')
    plt.legend()
    plt.show()
    #Gráfica iteraciones, x1 vs x2
    plt.figure(figsize=(10, 6))
    plt.title(f"Solución Aproximada vs Solución Real (h={h:.4f})")
    plt.plot(tabla_datos['x1_n'], tabla_datos['x2_n'], label='Solución Aproximada', color="g")
    plt.scatter(tabla_datos['x1_n'], tabla_datos['x2_n'], color="g") 
    plt.plot(x1_real, x2_real, label='Solución Real', color="orange")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    
# Tabla de errores
errores_x2 = errores["error_x2"]
errores["ln(error_x2)"] = []
for error_x2 in errores_x2:
    errores["ln(error_x2)"].append(log(error_x2))

tabla_errores = pd.DataFrame(errores)
tabla_datos.to_excel(archivo_excel, sheet_name=f'errores', index=False)

# Graficar h vs ln(error)
plt.figure(figsize=(10, 6))
plt.plot(tabla_errores['h'], tabla_errores['ln(error_x2)'], color="blue")
plt.scatter(tabla_errores['h'], tabla_errores['ln(error_x2)'], color="blue")
plt.xlabel('h')
plt.ylabel('ln(error_x2)')
plt.title('h vs ln(error_x2)')
plt.grid(False)
plt.show()