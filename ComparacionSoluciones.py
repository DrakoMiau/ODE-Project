import numpy as np
from scipy.integrate import solve_ivp

# Nuevos parámetros del sistema de Lorenz
sigma = 10
rho = 28
beta = 8/3

# Define las ecuaciones del sistema de Lorenz
def lorenz(t, xyz):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Define el punto de equilibrio
x0 = np.sqrt(beta * (rho - 1))
y0 = np.sqrt(beta * (rho - 1))
z0 = rho - 1

# Nuevas condiciones iniciales
initial_conditions = [9 * np.sqrt(3), 9 * np.sqrt(3), rho - 1]

# Nuevos tiempos a evaluar
times = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.6, 1, 2]

# Resuelve el sistema no lineal en los tiempos especificados
sol = solve_ivp(lorenz, [0, max(times)], initial_conditions, t_eval=times)

# Función para el sistema linealizado
def linear_lorenz(t, xyz):
    x, y, z = xyz
    # Jacobiano en el punto de equilibrio
    jacobian = np.array([
        [-sigma, sigma, 0],
        [rho - z0, -1, -x0],
        [y0, x0, -beta]
    ])
    dxdt, dydt, dzdt = np.dot(jacobian, xyz - [x0, y0, z0])
    return [dxdt, dydt, dzdt]

# Resuelve el sistema linealizado en los tiempos especificados
sol_linear = solve_ivp(linear_lorenz, [0, max(times)], initial_conditions, t_eval=times)

# Imprime los valores obtenidos para comparar
for i, t in enumerate(times):
    print(f"Valores en t={t} del sistema no lineal:", sol.y[:, i])
    print(f"Valores en t={t} del sistema linealizado:", sol_linear.y[:, i])
