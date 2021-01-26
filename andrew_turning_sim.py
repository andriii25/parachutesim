from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.integrate import solve_ivp
import ambiance
import matplotlib.pyplot as plt
import numpy as np
from andrew_sim import cd, mach
from plotting.piticks import pi_axis_formatter

def fin_drag_coeff(theta):
    theta_norm_remainder = int(theta // (np.pi / 2))
    theta_norm = theta % (np.pi / 2)
    if theta_norm_remainder % 2 == 1:
        # Is even?
        theta_norm = np.pi / 2 - theta_norm
    drag_coeff = 2 * (np.sin(theta_norm) ** 4) + 2 * (np.sin(theta_norm) ** 3) * np.cos(theta_norm)
    return drag_coeff


def drag(v, h, S, drag_coeff):
    density = ambiance.Atmosphere(h).density
    return 0.5 * drag_coeff * density * (v ** 2) * S


def drag_parachute(v, h):
    #S = 0.2425
    S = 1
    drag_coeff = cd(mach(v, h))
    return drag(v, h, S, drag_coeff)


def drag_body(theta, v, h):
    S = 9.5e-3 * np.cos(theta)
    drag_coeff = -1
    # return drag(v, h, S, drag_coeff)
    return 0


def drag_fin(theta, v, h):
    S = 0.03
    drag_coeff = fin_drag_coeff(theta)
    return drag(v, h, S, drag_coeff)


def angular_acceleration(ang, ang_vel, v, h):
    mom_of_inertia = 4
    parachute_arm = 0.95
    body_arm = 0.1
    fin_arm = 0.95

    ang_acc = 1 / mom_of_inertia * (np.sin(ang) * parachute_arm * 13.1719 + # drag_parachute(v, h) -
                                    #0 * np.sin(ang) * body_arm * drag_body(ang, v, h) -
                                    #0 * fin_arm * drag_fin(ang, v, h) +
                                    (-0.5) * (ang_vel ** 3) )

    return ang_acc

def f(t, th, v, h):
    theta_dot = th[0]
    theta = th[1]

    angular_acc = angular_acceleration(theta, theta_dot,  v, h)
    angular_vel = theta_dot

    return np.array([angular_acc, angular_vel])


def visualise(soln, v, h):
    """Plots all the data from the solution"""
    # Generally if you want to add an other plot make a function for it like drag() or acc()
    # And recalculate from the solution, look at the examples for drag

    # Set fonts
    plt.rcParams.update({'font.size': 18})


    t = soln.t
    ang_vel = soln.y[0]
    theta = soln.y[1]
    ang_acc = [angular_acceleration(theta_curr, omega_curr, v, h) for theta_curr, omega_curr in zip(theta, ang_vel)]
    # drag = [drag_parachute(v, h, 0.2425) for v, h, in zip(v, x)]

    print(f'Max angular acceleration: {np.max(ang_acc)}')
    print(f'Max angular velocity: {np.max(abs(ang_vel))}')
    print(f'Max angle: {np.max(theta)}')
    #print(f'Max parachute drag: {np.max(drag)}')
    #print(f'Velocity at end: {v[-1]} m/s')

    fig, ax = plt.subplots(1, 1, figsize=(18, 6), sharex="col")

    ticklen = np.pi/4
    ax.set_title("Angle of the rocket against the vertical over time")
    ax.plot(t, theta, 'g-', label="Angle")
    ax.set_ylabel('$\Theta$  (radians)')
    ax.set_ylim(top=1.7*np.pi)
    ax.set_xlabel('time $(s)$')
    ax.yaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
    ax.yaxis.set_major_locator(MultipleLocator(base=ticklen))
    ax.margins(0)
    ax.legend()

    # ax[1].plot(t, abs(ang_vel), 'b-', label="Angular velocity")
    # ax[1].set_ylabel('$\dot{\Theta}\quad(\frac{1}{s})$')
    # ax[1].legend()
#
#
#     ax[2].plot(t, ang_acc, 'r-', label="Angular acceleration")
#     ax[2].set_ylabel('$\ddot{\Theta}\quad(\frac{1}{s^2})$')
#    ax[2].legend()
    #ax[3].plot(t, drag, 'r-', label="Parachute drag")
    #ax[3].set_ylabel('Parachute drag (N)')
    #ax[3].legend()

    #ax[4].plot(t, m, 'y-', label="Mach number")
    #ax[4].set_ylabel('Mach number (-)')
    #ax[4].legend()

    plt.show()
def plot_drag_coeff_test():
    angles = np.linspace(0, 2 * np.pi, 10001)


    test = [(2 * (np.sin(x) ** 4) + 2 * (np.sin(x) ** 3) * np.cos(x)) for x in angles]
    drag_coeff = [fin_drag_coeff(th) for th in angles]
    plt.plot(angles, drag_coeff)
    plt.show()

def turned_around(t, theta, v, h):
    return theta[1] - 2 * np.pi

if __name__ == "__main__":
    t_span = np.array([0, 20])
    times = np.linspace(t_span[0], t_span[1], 1001)

    plot_drag_coeff_test()

    #ThetaDot0, Theta0, theta measured as angle to the vertical
    y0 = [0, 0.1]
    v = 1000#TODO: Get actual
    h = 50000#TODO: Get actual

    turned_around.terminal = False
    turned_around.direction = 1

    soln = solve_ivp(f, t_span, y0, dense_output=True, args=[v, h], method="Radau", events=turned_around, t_eval=times)

    visualise(soln, v, h)

