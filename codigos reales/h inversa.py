from sympy import symbols, solve, Matrix

# Definir símbolos y h
t_n, h = symbols('t_n h')

# Definir los valores de fn, fn1 y fn2
f_n = t_n**3
f_n1 = (t_n - h)**3
f_n2 = (t_n - 2*h)**3

# Definir la matriz inversa M
M_inv = Matrix([[t_n**2, t_n, 1],
                [(t_n**2 - 2*t_n*h + h**2), (t_n - h), 1],
                [(t_n**2 - 4*t_n*h + 4*h**2), (t_n - 2*h), 1]]).inv()

# Definir el vector b
b = Matrix([[f_n], [f_n1], [f_n2]])

# Multiplicar la inversa de M por el vector b para obtener los coeficientes A, B, y C
resultado = M_inv * b

# Resolver el sistema de ecuaciones para A, B, y C
A, B, C = symbols('A B C')
sistema_ecuaciones = [
    A - resultado[0],
    B - resultado[1],
    C - resultado[2]
]

# Resolver el sistema de ecuaciones
solucion = solve(sistema_ecuaciones, (A, B, C))

print("Los coeficientes A, B y C en términos de h son:")
print(solucion)
