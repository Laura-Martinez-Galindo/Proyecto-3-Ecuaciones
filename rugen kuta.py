from math import cos, sin
from tabulate import tabulate

def f(t, x1, x2):
    return 9 * x1 + 24 * x2 + 5 * cos(t) - 1/3 * sin(t)

def g(t, x1, x2):
    return -24 * x1 - 51 * x2 - 9 * cos(t) + 1/3 * sin(t)

def rk4(t0, y0, tn, h):
    n = int((tn - t0) / h)
    table = []
    
    for i in range(n):
        k1_x1 = h * f(t0, y0[0], y0[1])
        k1_x2 = h * g(t0, y0[0], y0[1])
        
        k2_x1 = h * f(t0 + h/2, y0[0] + h/2 * k1_x1, y0[1] + h/2 * k1_x2)
        k2_x2 = h * g(t0 + h/2, y0[0] + h/2 * k1_x1, y0[1] + h/2 * k1_x2)
        
        k3_x1 = h * f(t0 + h/2, y0[0] + h/2 * k2_x1, y0[1] + h/2 * k2_x2)
        k3_x2 = h * g(t0 + h/2, y0[0] + h/2 * k2_x1, y0[1] + h/2 * k2_x2)
        
        k4_x1 = h * f(t0 + h, y0[0] + h * k3_x1, y0[1] + h * k3_x2)
        k4_x2 = h * g(t0 + h, y0[0] + h * k3_x1, y0[1] + h * k3_x2)
        
        x1n = y0[0] + (k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1) / 6
        x2n = y0[1] + (k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2) / 6
        
        table.append((i, round(t0, 2), (round(x1n, 2), round(x2n, 2)), (round(k1_x1, 2), round(k1_x2, 2)),
                      (round(k2_x1, 2), round(k2_x2, 2)), (round(k3_x1, 2), round(k3_x2, 2)),
                      (round(k4_x1, 2), round(k4_x2, 2))))
        y0 = [x1n, x2n]
        t0 += h
    
    return table

# Entradas
t0 = 0
y0 = [4/3, 2/3]
tn = 20
h_values = [0.2, 0.1] + [2**(-k)/15 for k in range(5)]

for h in h_values:
    print(f"\nCon h = {h:.4f}:")
    result = rk4(t0, y0, tn, h)
    headers = ["n", "tn", "(x1n, x2n)", "k1n", "k2n", "k3n", "k4n"]
    print(tabulate(result, headers=headers, tablefmt="grid"))
