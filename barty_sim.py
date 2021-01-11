from ambiance import Atmosphere
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

non_ambiance_alt = {90000: 2.85e-6, 95000: 9.91e-7, 100000: 3.73e-7, 105000: 1.54e-7, 110000: 5.93e-8, 115000: 2.71e-8,
                    120000: 1.48e-8}  # measured high altitude densities from attached paper
heights = np.linspace(1000, 120000, 120)
density = np.zeros(120)

for i in range(len(heights)):
    if heights[i] <= 81000:
        atmosphere = Atmosphere(heights[i])
        density[i] = atmosphere.density
    elif heights[i] < 90000:
        atmosphere = Atmosphere(81000)
        density[i] = atmosphere.density
    else:
        rounded_h = round(heights[i] / 5000) * 5000  # Rounding and approximating as high altitude data
        atmosphere = non_ambiance_alt[rounded_h]


def current_density(height):
    density_c = np.interp(height, heights, density)
    return density_c


def Mach(height, velocity):
    if height <= 81000:
        M = velocity / Atmosphere(height).speed_of_sound
        return M
    else:
        M = velocity / Atmosphere(81000).speed_of_sound
        return M


def CD(M):
    CD = 0.572 - 0.08 * M
    if CD < 0.42:
        return CD
    else:
        return 0.42


m = 10  # kg
g = 9.81  # ms^-2
S = 0.2425  # m^2
W = m * g
index = 0  # for looping
dt = 0.1  # change this for performance, default 0.1s, gets unsteady over about 5s

h = []
h.append(120000)

v = []
v.append(0)

a = []
a.append(-9.81)

dp = []
dp.append(0)

Dt = []
Dt.append(0)

Bodydrag = []
Bodydrag.append(0)

index = 0
current_height = h[0]
time = []
time.append(0)
while current_height > 1000:
    M = Mach(h[index], v[index])

    D = 0.5 * CD(M) * current_density(h[index]) * S * (v[index]) ** 2
    D_body = 0.5 * 1.14 * current_density(h[index]) * 9.5e-3 * (v[index]) ** 2  # Rough approx for bluff body drag
    dp1 = current_density(h[index]) * 1 / 2 * (v[index]) ** 2
    accel = (D + D_body - W) / m
    a.append(accel)

    v_new = v[index] - a[index] * dt
    current_height = h[index] - v_new * dt

    v.append(v_new)
    h.append(current_height)
    time.append(time[index] + 0.1)
    dp.append(dp1)
    Dt.append(D)
    Bdrag = (1 / 2 * current_density(h[index]) * (v[index]) ** 2 * 0.03) * (2)
    Bodydrag.append(Bdrag)

    index += 1

    if index > 100000:
        break

plot_accel = [(x / 9.81) for x in a]

plt.plot(time, h)
plt.xlabel('$Time /s$')
plt.ylabel('$Height /m$')
plt.show()

plt.plot(time, v)
plt.xlabel('$Time /s$')
plt.ylabel('$Velocity /ms^-1$')
plt.show()

plt.plot(time, plot_accel)
plt.xlabel('$Time /s$')
plt.ylabel('$Accel /g$')
plt.show()

plt.plot(h, dp)
plt.xlabel('$Height /m$')
plt.ylabel('$Dynamic Pressure N/m^2$')
plt.show()

plt.plot(h, Dt)
plt.xlabel('$Height /m$')
plt.ylabel('$Drag N/m^2$')
plt.show()

print(v[-1:])
n = 0
max = 0
for i in range(len(a)):
    if a[i] > max:
        max = a[i]
    n = i
alt = h[n]
mac = a[n]
print(mac, alt)

plt.plot(h, Dt, label='Parachute Drag')
plt.plot(h, Bodydrag, label='Findrag')
plt.xlim(0)
plt.ylim(0)
plt.xlabel('$Height /m$')
plt.ylabel('$Drag N$')
plt.legend()
plt.show()