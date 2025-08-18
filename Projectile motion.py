import math
import numpy as np
import matplotlib.pyplot as plt
#A basic projectile motion calculato/ animatio

Speed = float(input("Enter speed in m/s: "))
Angle = float(input("Enter angle in degrees: "))
Angle_rad = Angle * 3.14 / 180
Vel_x = Speed * math.cos(Angle_rad)
Vel_y = Speed * math.sin(Angle_rad)
t_max = (2 * Vel_y) / 10
t = np.linspace(0 , t_max ,100)
Dist_x = Vel_x * t
Dist_y = Vel_y * t - (0.5 * 10 * t * t)
plt.plot(Dist_x, Dist_y)
plt.xlabel("Horizontal Distance")
plt.ylabel("Vertical Distance")
plt.title("Projectile motion")
plt.grid(True)
plt.show()


