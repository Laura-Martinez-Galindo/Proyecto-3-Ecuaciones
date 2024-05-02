from sympy import symbols, solve

# Definimos las variables
A, B, C, D = symbols('A B C D')

# Definimos las ecuaciones del sistema
eq1 = -A - 9*C - 24*D + 1/3
eq2 = C - 9*A - 24*B - 5
eq3 = -B + 24*C + 51*D - 1/3
eq4 = D + 24*A + 51*B + 9

# Resolvemos el sistema de ecuaciones
solution = solve((eq1, eq2, eq3, eq4), (A, B, C, D))

# Mostramos los resultados
print("Soluciones:")
print("A =", solution[A])
print("B =", solution[B])
print("C =", solution[C])
print("D =", solution[D])
