from sympy import symbols, Eq, solve

# Definir las variables
x, y, z, w = symbols('x y z w')

# Definir las constantes como símbolos abstractos
a, b, c, d, f, g, h, j = symbols('a b c d f g h j')

# Definir las ecuaciones
eq1 = Eq(a**3 * x + a**2 * y + a * z + w, f)
eq2 = Eq(b**3 * x + b**2 * y + b * z + w, g)
eq3 = Eq(c**3 * x + c**2 * y + c * z + w, h)
eq4 = Eq(d**3 * x + d**2 * y + d * z + w, j)

# Resolver el sistema de ecuaciones
sol = solve((eq1, eq2, eq3, eq4), (x, y, z, w))

# Imprimir la solución
print("Solución:")
print("x =", sol[x])
print("y =", sol[y])
print("z =", sol[z])
print("w =", sol[w])
