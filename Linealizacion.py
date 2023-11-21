import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del sistema de Lorenz
sigma = 10
rho = 28
beta = 8/3

# Definición de las ecuaciones del sistema de Lorenz
def lorenz(t, xyz):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

x0 = np.sqrt(beta * (rho - 1))
y0 = np.sqrt(beta * (rho - 1))
z0 = rho - 1

# Solución del sistema no lineal
initial_conditions = [1.0, 1.0, 1.0]
sol = solve_ivp(lorenz, [0, 50], initial_conditions, t_eval=np.linspace(0, 50, 5000))


def jacobian_lorenz(x, y, z):
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])

# Se hace el calculo del jacobiano en el punto de equilibrio
jac_at_eq = jacobian_lorenz(x0, y0, z0)

# soluciona el sistema al rededor del punto de equilibrio
def linear_lorenz(t, xyz):
    x, y, z = xyz
    dxdt, dydt, dzdt = np.dot(jac_at_eq, xyz - [x0, y0, z0])
    return [dxdt, dydt, dzdt]

initial_conditions_perturbed = [x0 + 0.1, y0 + 0.1, z0 + 0.1]
sol_linear = solve_ivp(linear_lorenz, [0, 50], initial_conditions_perturbed, t_eval=np.linspace(0, 50, 5000))

# Graficar las soluciones
fig = plt.figure(figsize=(12, 6))

# Grafica del sistema no lineal
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(sol.y[0], sol.y[1], sol.y[2], label='No lineal')
ax1.set_title('Atractor de Lorenz (No lineal)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Grafica del sistema linealizado
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(sol_linear.y[0], sol_linear.y[1], sol_linear.y[2], label='Linealizado')
ax2.set_title('Atractor de Lorenz Linealizado')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()
