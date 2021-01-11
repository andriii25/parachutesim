import numpy as np
import matplotlib.pyplot as plt
import ambiance
from scipy.integrate import solve_ivp


def cd(m):
    """Returns drag coefficient of parachute as a function of mach number m"""
    return min(0.42, 0.572 - 0.08 * abs(m))


def mach(v, h):
    """Returns mach number of velocity v at a certain height h"""
    # Approximates
    return v / ambiance.Atmosphere(h).speed_of_sound if h < 81000 else v / ambiance.Atmosphere(81000).speed_of_sound


# TODO: Maybe one general drag function?
def drag_parachute(v, h, S):
    """Returns drag force for the parachute as a function of velocity, height and surface area S"""
    if h > 81000:
        return 0
    drag_coeff = cd(mach(v, h))
    density = ambiance.Atmosphere(h).density
    return 0.5 * drag_coeff * density * (v ** 2) * S


def drag_body(v, h):
    """Returns drag force for the body as a function of velocity, height and surface area S"""
    if h > 81000:
        return 0
    drag_coeff = 1.14  # Estimate from Ben's code?
    density = ambiance.Atmosphere(h).density
    S = 9.5e-3  # Estimate from OpenRocket

    return 0.5 * drag_coeff * density * (v ** 2) * S


def acc(v, h, S):
    g = 9.81
    m = 10

    a = (drag_parachute(v, h, S) + drag_body(v, h) - m * g) / m
    return a


def f(t, y, S):
    v = y[0]
    h = y[1]

    dvdt = acc(v, h, S)
    dxdt = v

    return np.array([dvdt, dxdt], dtype="object")


def visualise(soln):
    """Plots all the data from the solution"""
    # Generally if you want to add an other plot make a function for it like drag() or acc()
    # And recalculate from the solution, look at the examples for drag
    t = soln.t
    v = soln.y[0]
    x = soln.y[1]
    a = [acc(v, h, 0.2425) / 9.81 for v, h in zip(v, x)]
    m = [abs(mach(v, h)) for v, h in zip(v, x)]
    drag = [drag_parachute(v, h, 0.2425) for v, h, in zip(v, x)]

    print(f'Max acceleration: {np.max(a)}')
    print(f'Max velocity: {np.max(abs(v))}')
    print(f'Max parachute drag: {np.max(drag)}')
    print(f'Velocity at end: {v[-1]} m/s')

    fig, ax = plt.subplots(5, 1, figsize=(18, 18), sharex="col")

    ax[0].plot(t, abs(v), 'b-', label="Velocity")
    ax[0].set_ylabel('v (m/s)')
    ax[0].legend()

    ax[1].plot(t, x, 'g-', label="Position")
    ax[1].set_ylabel('x $(m)$')
    ax[1].set_xlabel('time $(s)$')
    ax[1].legend()

    ax[2].plot(t, a, 'r-', label="Acceleration")
    ax[2].set_ylabel('a $(m/s^2)$')
    ax[2].legend()
    ax[3].plot(t, drag, 'r-', label="Parachute drag")
    ax[3].set_ylabel('Parachute drag (N)')
    ax[3].legend()

    ax[4].plot(t, m, 'y-', label="Mach number")
    ax[4].set_ylabel('Mach number (-)')
    ax[4].legend()

    plt.show()


def hit_goal(t, y, S):
    """Function that gives 0 when rocket is at the required height"""
    return y[1] - 1000


if __name__ == "__main__":
    t_span = np.array([0, 1000])
    times = np.linspace(t_span[0], t_span[1], 101)

    y0 = np.array([0, 120000])

    # The event needed so that it stops solving when height is at 1000 m
    hit_goal.terminal = True
    hit_goal.direction = -1  # end when hit_goal positive --> negative

    # TODO: Do a surface sweep like Ellis' code (will probably go rather slow)
    # test_areas = np.linspace(0, 1, 1001)
    # count = 0;
    # for S in test_areas:
    #    count += 1
    #    print(f'{count/test_areas.size:.1%} done', end='\r')
    #
    # print("")

    soln = solve_ivp(f, t_span, y0, dense_output=True, events=hit_goal, method='LSODA', args=[0.2425])
    visualise(soln)
