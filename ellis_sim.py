from ambiance import Atmosphere
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

heights = np.linspace(1000, 81000, 81)
atmosphere = Atmosphere(heights)
density = atmosphere.density


def CD(M):
    return np.array([min(0.42, (0.572 - 0.08 * m)) for m in M])


S_size = 1000  # change for performance, default 1000
Sfin = 0.03
S = np.linspace(0.0005, 1.0005, S_size)  # avoid division by 0
m = 10  # kg
g = 9.81  # ms^-2
h = 81000 * np.ones(S_size)  # representing h,v,a at time t
v = 875 * np.ones(S_size)  #
a = np.zeros(S_size)  #
t = 0  # init t
index = 0  # for looping
dt = 0.1  # change this for performance, default 0.1s, gets unsteady over about 5s
t_size = int(400 / dt)  # leave 1000 be, it makes sure we reach steady state at 6km
# (S0,time)
drag = np.zeros((S_size, t_size))  # array representing drag on parachute as a function of both S and t
hei = np.zeros((S_size, t_size))  # array representing height as a function of both S and t
vel = np.zeros((S_size, t_size))  # array representing velocity as a function of both S and t
accel = np.zeros((S_size, t_size))  # array representing acceleration as a function of both S and t
mach = np.zeros((S_size, t_size))  # array representing mach number as a function of both S and t
dpres = np.zeros((S_size, t_size))  # array representing dynamic pressure bla bla
dragfin = np.zeros((S_size, t_size))  # drag on the fins
n = S_size * t_size
print(
    f'Computing with {S_size} values for S from 0 to 1, {int(1000 / dt)} values for t from 0 to 1000, n={S_size * t_size}...')
time0 = time.time()
while index < t_size:
    if index % (t_size * 0.01) == 0:
        delta = time.time() - time0

        pc = int(1 + 100 * index / t_size)

        remaining = int(delta * (100 - pc) / pc)
        print(f'> {pc}% complete, {remaining}s remaining', end='\r')# UI
    M = v / Atmosphere(h).speed_of_sound

    D = CD(M) * (S) * np.interp(h, heights,
                                density) * 0.5 * v ** 2  # C_D_0 * S{no modelling for fuselage drag} * rho * 0.5 * v^2
    # you can add a +0.7 to S in the above line to approximate fuselage drag but actually it will greatly increase around M=1-2 so it does need refining
    W = m * g

    a = (D - W) / m

    v -= a * dt  # integrations
    h -= v * dt

    for i, x in enumerate(h):
        if x < 0: h[i] = 0
    dpres[:, index] = np.interp(h, heights, density) * 0.5 * v ** 2
    drag[:, index] = D  # assign columns of values to matrices
    dragfin[:, index] = 2 * Sfin * np.interp(h, heights, density) * 0.5 * v ** 2
    hei[:, index] = h
    vel[:, index] = v
    accel[:, index] = a
    mach[:, index] = M
    t += dt
    index += 1
print("")
time = np.linspace(0, t, t_size)
max_accel_indices = np.zeros(S_size, dtype=int)
max_accel_vals = np.zeros(S_size)
print('calculating velocities at 6km...')
end_vels = [np.interp(1000, np.linspace(0, 1000, t_size), v) for v in vel]  # evaluate velocity at 6000m
print('calculating maximums...')
for r, row in enumerate(accel):  # a real ugly way to make the maximums lists
    maximum = 0
    index = 0
    for i, val in enumerate(row):
        if val > maximum:
            maximum = val
            index = i

    max_accel_indices[r] = int(index)
    max_accel_vals[r] = maximum

print('finding heights at maximum accelerations...')
max_accel_heights = [hei[i, max_accel_indices[i]] for i in
                     range(S_size)]  # find height at which max acceleration occurs
accel /= g  # normalise to gs
'''
# i'm trying to make an interactive version of ax1 here where you can vary S_0
def update(val):
  i=np.where(S==A.val)
  h=hei[i]
  a=accel[i]
  ax.set_xdata(h)
  ax.set_ydata(a)
  fig.canvas.draw_idle()
fig, ax = plt.subplots()
axamp = plt.axes([0.25, 0.1, 0.65, 0.03])
A = Slider(axamp, 'S$_0$', 0, 1, valinit=0.5)
i=np.where(S==A.val)
h=hei[i]
a=accel[i]
ax.plot(h,a)
A.on_changed(update)
plt.show() 
'''
print('plotting...')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)  # plotting
# ACCEL/HEIGHT at S=0.56 hei[int(0.27*S_size)]
ax1.plot(hei[int(0.265 * S_size)], drag[int(0.265 * S_size)], label="parachute")  # .265
ax1.plot(hei[int(0.265 * S_size)], dragfin[int(0.265 * S_size)], label="fins")
ax1.set_ylabel('Drag Force (N)')
ax1.legend()
# ax1.set_xlim(80000,0)
ax1.set_title('Drag Force/Time')
# VELOCITY/HEIGHT at S=0.56
ax3.plot(hei[0], mach[int(0 * S_size)])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Mach (M)')
# ax3.set_xlim(80000,0)
ax3.set_title('Mach/Time')
# MAX_DRAG/S0
ax2.plot(S, max_accel_vals / g)
# ax2.plot(S,end_vels)
ax2.set_ylabel('Max Acceleration (g)')
ax2.set_title('Max Acceleration/S$_0$')
# HEIGHTOF/S0
ax4.plot(S, end_vels)
ax4.set_xlabel('S$_0$ (m$^2$)')
ax4.set_ylabel('Velocity at 1km (ms$^{-1}$)')
ax4.set_title('Velocity at 1km/S$_0$')
plt.show()
