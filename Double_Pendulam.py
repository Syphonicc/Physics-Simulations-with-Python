import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib

# Use TkAgg backend to ensure a popup window appears on Windows 11/PyCharm
matplotlib.use('TkAgg')

t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1, L2')

the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
#Explicitly writing them as functions of time t
the1 = the1(t)
the2 = the2(t)
#Defining derivatives and second derivatives
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)
#Defining x1,y1,x2,y2
x1 = L1 * smp.sin(the1)
y1 = -L1 * smp.cos(the1)
x2 = L1 * smp.sin(the1) + L2 * smp.sin(the2)
y2 = -L1 * smp.cos(the1) - L2 * smp.cos(the2)
#Defining Kinetic and Potential energy of each mass

#Kinetic
T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1 + T2
#Potential
V1 = m1 * g * y1
V2 = m2 * g * y2
V = V1 + V2
#Lagrangian
L = T - V
# Getting the LE1 and LE2 for 2 equations of motions
LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t)
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t)
#Solving Lagranges equations(this assumes that LE1 and LE2 are both equal to zero)
sols = smp.solve([LE1, LE2], (the1_dd, the2_dd),
                 simplify=False, rational=False)
#In python we can only solve systems of First order ODEs so we need to convert 2nd Order to First order
#We will assign z1 = d0/dt then dz1/dt = d^20/dt^2
#To convert a symbolic expression into a numerical function we use lambdify

dz1dt_f = smp.lambdify((t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the1_dd])
dz2dt_f = smp.lambdify((t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the2_dd])

#Now we need define a "State-Vector" S it makes it easy for the computer to predict the next move and calculate faster
#S = (01,z1,02,z2)
def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        z1,
        dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        z2,
        dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
    ]

#We can now solve the ODE using odeint
t_vals = np.linspace(0, 40, 1001)
g_val = 9.81
m1_val = 2
m2_val = 1
L1_val = 2
L2_val = 1
ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t_vals, args=(g_val, m1_val, m2_val, L1_val, L2_val))

#Now we can obtain 01 and 02 from the answer
the1_vals = ans[:, 0]
the2_vals = ans[:, 2]

x1_p = L1_val * np.sin(the1_vals)
y1_p = -L1_val * np.cos(the1_vals)
x2_p = x1_p + L2_val * np.sin(the2_vals)
y2_p = -L1_val * np.cos(the1_vals) - L2_val * np.cos(the2_vals)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])  # enable this to hide x axis ticks
ax.get_yaxis().set_ticks([])  # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
ax.set_ylim(-4, 4)
ax.set_xlim(-4, 4)

def animate(i):
    ln1.set_data([0, x1_p[i], x2_p[i]], [0, y1_p[i], y2_p[i]])
    return ln1,

ani = animation.FuncAnimation(fig, animate, frames=1000, interval=40, blit=True)
plt.show()