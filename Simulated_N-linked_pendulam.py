import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = int(input("Enter number of links: "))
L = np.ones(N)  # Length of each link (1 meter)
M = np.ones(N)  # Mass of each mass (1 kg)
G = 9.81  # Gravity


initial_thetas = np.linspace(np.pi / 2, np.pi / 4, N)
initial_omegas = np.zeros(N)
state0 = np.concatenate([initial_thetas, initial_omegas])



def get_derivs(t, state):

    theta = state[:N]
    omega = state[N:]





    A = np.zeros((N, N))
    B = np.zeros(N)

    for i in range(N):

        B[i] -= G * np.sin(theta[i]) * np.sum(M[i:])

        for j in range(N):

            mass_sum = np.sum(M[max(i, j):])
            A[i, j] = mass_sum * L[i] * L[j] * np.cos(theta[i] - theta[j])


            B[i] -= mass_sum * L[i] * L[j] * np.sin(theta[i] - theta[j]) * omega[j] ** 2


    alpha = np.linalg.solve(A, B)


    return np.concatenate([omega, alpha])



t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)
sol = solve_ivp(get_derivs, t_span, state0, t_eval=t_eval, method='RK45')


fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-N - 0.5, N + 0.5)
ax.set_ylim(-N - 0.5, N + 0.5)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2, color='#2c3e50', markersize=8)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def update(frame):
    current_thetas = sol.y[:N, frame]


    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    for i in range(N):
        x[i + 1] = x[i] + L[i] * np.sin(current_thetas[i])
        y[i + 1] = y[i] - L[i] * np.cos(current_thetas[i])

    line.set_data(x, y)
    time_text.set_text(f't = {sol.t[frame]:.1f}s')
    return line, time_text


ani = FuncAnimation(fig, update, frames=len(sol.t), interval=20, blit=True)
plt.title(f"Simulated {N}-Linked Pendulum")
plt.grid(alpha=0.3)
plt.show()