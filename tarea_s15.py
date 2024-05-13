import pandas as pd 
import numpy as np

def bn(a, n):
    res= - (( a/(np.pi*n))**3)*(np.pi*n*np.sin(np.pi*n)+2*np.cos(np.pi*n)-2) 
    return res

def u_n(a, n, x, y):
    b_n=bn(a,n)
    u=b_n* np.sin(n*np.pi*x/a)*(np.e**(-n*np.pi*y/a))*2/a
    return u
def f(a,x,y):
    u=0
    for n in range (1,1000):
        u+=u_n(a,n,x,y)
    return u
from scipy.optimize import newton

# Definimos la función para la cual queremos encontrar la raíz
def func(y0, a, x):
    return f(a, x, y0) - 0.1

# Definimos la derivada de la función
def func_deriv(y0, a, n, x):
    return n*np.pi/a * u_n(a, n, x, y0)

# Parámetros
a = 5
for x in np.linspace(1,5,1000):
    # Usamos el método de Newton para encontrar y0
    y0_initial_guess = 1
    y0 = newton(func, y0_initial_guess, args=(a, x))
    print(f(a,x,y0))
    print(f"El valor mínimo de y0 es {y0} para x={x}")
