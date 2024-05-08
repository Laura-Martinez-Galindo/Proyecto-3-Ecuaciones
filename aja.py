import numpy as np
import matplotlib.pyplot as plt

def gamma(x, y):
    return 3*(1-x)*2 * np.exp((-1)*(x*2)-(y-1)*2) - 10*((x/5)-(x*3)-(y*5))*np.exp((x*-2)-(y*2)) - (1/3)*np.exp((-(x+1)*2)-(y*2))

def gamma2(x, y):
    return (-1)*(3*(1-x)*2 * np.exp((-1)*(x*2)-(y-1)*2) - 10*((x/5)-(x*3)-(y*5))*np.exp((x*-2)-(y*2)) - (1/3)*np.exp((-(x+1)*2)-(y*2)))

def descenso_gradiente(x0, tasa_aprendizaje, tolerancia, iteraciones_max, f):
    k = 0
    lista_puntos = [x0]
    gradiente = np.zeros_like(x0)
    norma = np.linalg.norm(gradiente)

    while norma >= tolerancia and k < iteraciones_max:
        gradiente = calcular_gradiente(f, lista_puntos[k])
        norma = np.linalg.norm(gradiente)
        lista_puntos.append(lista_puntos[k] - tasa_aprendizaje * gradiente)
        k += 1

    return lista_puntos

def calcular_gradiente(f, x):
    h = 1e-6  # Pequeño incremento para calcular derivadas numéricas
    gradiente = np.zeros_like(x)

    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        gradiente[i] = (f(x_plus_h) - f(x)) / h

    return gradiente

# Puntos iniciales
x0_list = [(-0.5, -1.2), (-2, -1), (1, 2), (2, 1)]

def graficar_puntos(X, titulo):
    x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = gamma(x1, x2)
    
    plt.contourf(x1, x2, Z, levels=20, cmap='coolwarm', alpha=0.6)  # Agregar sombreado de profundidad
    plt.contour(x1, x2, Z, levels=20, colors='k', alpha=0.8)
    
    x_coords = [point[0] for point in X]
    y_coords = [point[1] for point in X]

    plt.scatter(x_coords, y_coords, s=15)
    plt.plot(x_coords, y_coords, '-o', markersize=3)  # Cambiar a '-o' para marcar puntos
    plt.title(titulo)
    plt.show()

# Realizar descenso de gradiente y graficar para cada punto inicial (mínimos)
for x0 in x0_list:
    result_min_points = descenso_gradiente(x0, 0.3, (10)**-5, 1000000, gamma)
    graficar_puntos(result_min_points, f"Mínimo: α = 0.3, x0 = {x0}, Mínimo: {result_min_points[-1]}")

# Realizar descenso de gradiente y graficar para cada punto inicial (máximos)
for x0 in x0_list:
    result_max_points = descenso_gradiente(x0, 0.3, (10)**-5, 1000000, gamma2)
    graficar_puntos(result_max_points, f"Máximo: α = 0.3, x0 = {x0}, Máximo: {result_max_points[-1]}")

'''def descenso_gradiente(x0, tasa_aprendizaje, tolerancia, iteraciones_max, f):
    k = 0
    lista_puntos = [x0]
    gradiente = np.zeros_like(x0)
    norma = np.linalg.norm(gradiente)

    while norma >= tolerancia and k < iteraciones_max:
        gradiente = calcular_gradiente(f, lista_puntos[k])
        norma = np.linalg.norm(gradiente)
        lista_puntos.append(lista_puntos[k] - tasa_aprendizaje * gradiente)
        k += 1

    return lista_puntos

def calcular_gradiente(f, x):
    h = 1e-6  # Pequeño incremento para calcular derivadas numéricas
    gradiente = np.zeros_like(x)

    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        gradiente[i] = (f(x_plus_h) - f(x)) / h

    return gradiente

# Ejemplo de uso:
def funcion_ejemplo(x):
    return x[0]*2 + x[1]*2  # Función de ejemplo (puedes reemplazarla con tu propia función)

x0 = np.array([1.0, 2.0])  # Condición inicial
tasa_aprendizaje = 0.1
tolerancia = 1e-6
iteraciones_max = 100

lista_puntos = descenso_gradiente(x0, tasa_aprendizaje, tolerancia, iteraciones_max, funcion_ejemplo)
print("Puntos encontrados:")
for punto in lista_puntos:
    print(punto)'''