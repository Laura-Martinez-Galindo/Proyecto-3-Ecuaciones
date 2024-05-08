import sympy as sp

# Definir la matriz
A = sp.Matrix([[1, 1, 1],
               [0, -1, -2],
               [0, 1, 4]])

# Calcular la inversa
A_inv = A.inv()

print(A_inv)
