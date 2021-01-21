from scipy.integrate import solve_ivp
import ambiance
import matplotlib.pyplot as plt
import numpy as np
from andrew_sim import cd, mach


def drag(v, h, S, drag_coeff):
    density = ambiance.Atmosphere(h).density
    return 0.5 * drag_coeff * density * (v ** 2) * S


def drag_parachute(v, h):
    S = 0.2425 # TODO: Check?
    drag_coeff = cd(mach(v, h))
    return drag(v, h, S, drag_coeff)


def drag_body(theta, v, h):
    S = 9.5e-3 * np.cos(theta) #TODO: Get actual
    drag_coeff = -1    #TODO: Get actual
    # return drag(v, h, S, drag_coeff)
    return 0


def drag_fin(theta, v, h):
    S = -1 #TODO: Get actual
    drag_coeff = -1 #TODO: Get actual
    # return drag(v, h, S, drag_coeff)
    return 0


def angular_acceleration(ang, v, h):
    mom_of_inertia = 1  #TODO: Get actual
    parachute_arm = 0.5      #TODO: Get actual
    body_arm = 0.1         #TODO: Get actual
    fin_arm = 0.95           #TODO: Get actual

    ang_acc = np.sin(ang) / mom_of_inertia * (parachute_arm * drag_parachute(v, h) -
                                              body_arm * drag_body(ang, v, h) -
                                              fin_arm * drag_fin(ang, v, h))
    return ang_acc

def f(t, th, v, h):
    theta_dot = th[0]
    theta = th[1]

    angular_acc = angular_acceleration(theta, v, h)
    angular_vel = theta_dot

    return np.array([angular_acc, angular_vel])


def visualise(soln, v, h):
    """Plots all the data from the solution"""
    # Generally if you want to add an other plot make a function for it like drag() or acc()
    # And recalculate from the solution, look at the examples for drag
    t = soln.t
    ang_vel = soln.y[0]
    theta = soln.y[1]
    ang_acc = [angular_acceleration(theta_curr, v, h) for theta_curr in theta]
    # drag = [drag_parachute(v, h, 0.2425) for v, h, in zip(v, x)]

    print(f'Max angular acceleration: {np.max(ang_acc)}')
    print(f'Max angular velocity: {np.max(abs(ang_vel))}')
    print(f'Max angle: {np.max(theta)}')
    #print(f'Max parachute drag: {np.max(drag)}')
    #print(f'Velocity at end: {v[-1]} m/s')

    fig, ax = plt.subplots(3, 1, figsize=(18, 18), sharex="col")

    ax[0].plot(t, abs(ang_vel), 'b-', label="Angular velocity")
    ax[0].set_ylabel('omega (1/s)')
    ax[0].legend()

    ax[1].plot(t, theta, 'g-', label="Angle")
    ax[1].set_ylabel('theta $(1)$')
    ax[1].set_xlabel('time $(s)$')
    ax[1].legend()

    ax[2].plot(t, ang_acc, 'r-', label="Angular acceleration")
    ax[2].set_ylabel('alpha $(1/s^2)$')
    ax[2].legend()
    #ax[3].plot(t, drag, 'r-', label="Parachute drag")
    #ax[3].set_ylabel('Parachute drag (N)')
    #ax[3].legend()

    #ax[4].plot(t, m, 'y-', label="Mach number")
    #ax[4].set_ylabel('Mach number (-)')
    #ax[4].legend()

    plt.show()

def turned_around(t, theta, v, h):
    return theta[1] - 2 * np.pi

if __name__ == "__main__":
    t_span = np.array([0, 5])
    times = np.linspace(t_span[0], t_span[1], 1001)

    #ThetaDot0, Theta0, theta measured as angle to the vertical
    y0 = [0, 0.1]
    v = 600#TODO: Get actual
    h = 50000#TODO: Get actual

    turned_around.terminal = True
    turned_around.direction = 1

    soln = solve_ivp(f, t_span, y0, dense_output=True, args=[v, h], model = 'LSODA', events=turned_around, t_eval=times)

    visualise(soln, v, h)

