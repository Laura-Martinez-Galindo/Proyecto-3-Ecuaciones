from math import cos, sin, exp, log10
import sympy as sp

def solve_system(x1n, x2n, tn1, h):
    # Definimos las variables simbólicas
    x1n1, x2n1 = sp.symbols('x1n1 x2n1')

    # Definimos las ecuaciones
    eq1 = x1n1 - (x1n + h * (9*x1n1 + 24*x2n1 + 5*sp.cos(tn1) - (1/3)*sp.sin(tn1)))
    eq2 = x2n1 - (x2n + h * (-24*x1n1 - 51*x2n1 - 9*sp.cos(tn1) + (1/3)*sp.sin(tn1)))

    # Resolvemos el sistema de ecuaciones
    solution = sp.solve((eq1, eq2), (x1n1, x2n1))
    print(solution.keys())
    
    return solution[x1n1], solution[x2n1]

# Ejemplo de uso
x1n, x2n, tn1, h = 1, 2, 0.5, 0.1
result = solve_system(x1n, x2n, tn1, h)
print("Solución:")
print(result)