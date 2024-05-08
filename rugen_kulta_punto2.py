import math
import matplotlib.pyplot as plt
import numpy as np

def f(theta1, theta2, omega1_i, omega2_i):
    omega2_i_cuadrado = np.float64(omega2_i) ** 2
    omega1_i_cuadrado = np.float64(omega1_i) ** 2
    numerador = (-3 * np.sin(theta1) - np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * ((omega2_i_cuadrado) + (omega1_i_cuadrado) * np.cos(theta1 - theta2)))
    denominador = (3 - np.cos(2 * theta1 - 2 * theta2))
    f_funcion =  numerador/denominador
    return f_funcion

def g(theta1, theta2, omega1, omega2):
    omega2_i_cuadrado = np.float64(omega2) ** 2
    omega1_i_cuadrado = np.float64(omega1) ** 2
    return (2 * np.sin(theta1-theta2) * (2 * omega1_i_cuadrado + 2 * np.cos(theta1) + omega2_i_cuadrado * np.cos(theta1-theta2)))/(3 - np.cos(2*theta1-2*theta2))




def runge_kulta4(h, tf, caso):
    n = np.float64(tf) / np.float64(h)
    matriz_valores_iniciales=[[1,1,0,0],
                              [np.pi, 0,0,10**(-10)],
                              [2,2,0,0],
                              [2,2+10**(-3),0,0]]
    valores=[np.array(matriz_valores_iniciales[caso], dtype=np.float64)]
    theta2_values=[matriz_valores_iniciales[0][1]]
    for i in range (int(n)):
        theta1=valores[i][0] #x0
        theta2=valores[i][1] #y0
        omega1=valores[i][2] #vx0
        omega2=valores[i][3] #vy0
        k1=h*omega1
        l1=h*f(theta1, theta2, omega1, omega2)
        q1=h*omega2
        m1=h*g(theta1, theta2, omega1, omega2)
        k2= h * (omega1+k1/2)
        l2=h*f(theta1+k1/2, theta2+q1/2, omega1+l1/2, omega2+m1/2)
        q2=h*(omega2+q1/2)
        m2=h*g(theta1+k1/2,theta2+q1/2, omega1+l1/2, omega2+m1/2)
        k3=h*(omega1+k2/2)
        l3=h*f(theta1+k2/2, theta2+q2/2, omega1+l2/2, omega2+m2/2)
        q3=h*(omega2+q2/2)
        m3=h*g(theta1+k2/2, theta2+q2/2, omega1+l2/2, omega2+m2/2)
        k4=h*(omega1+k3)
        l4=h*f(theta1+k3, theta2+q3, omega1+l3, omega2+m3)
        q4=h*(omega2+q3)
        m4=h*g(theta1+k3, theta2+q3, omega1+l3, omega2+m3)
        n_omega1= omega1+ (l1+2*l2+2*l3+l4)/6
        n_omega2= omega2 + (m1+2*m2+2*m3+m4) / 6
        n_theta1=theta1 + (k1+2*k2+2*k3+k4) / 6
        n_theta2=  theta2 + (q1+2*q2+2*q3+q4) / 6
        nuevos=np.array([n_theta1, n_theta2, n_omega1, n_omega2], dtype=np.float64)
        theta2_values.append(n_theta2)
        valores.append(nuevos)
    return valores, theta2_values

plt.figure(figsize=(8, 6))
for caso in range (2):
    valores, theta2_values=runge_kulta4(0.05, 100, caso)
    t_values=np.linspace(0,100,len(theta2_values))
    
    plt.plot(t_values, theta2_values, label=r"$\theta_2$"+ f" para el caso de condiciones iniciales {caso+1}")
    plt.xlabel("Tiempo (t)")
    plt.ylabel(r"$\theta_2$")
    plt.title("Evolución de $\\theta_2$ en función del tiempo")
    plt.grid(True)
    plt.legend()
    plt.show()
