from functools import partial

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm

from boom_dynamics import boom_dynamics, fwd_kinematics, inv_kinematics
# from plotting_tools import animate_trajectory, plot_control_efforts, plot_joint_traj
from plotting_tools import plot_3x_trajectories, plot_3x_control_efforts, plot_3x_joint_traj
from plot2x import plot_2x_trajectories, plot_2x_control_efforts, plot_2x_joint_traj
from PD_gen import generate_PD_trajectories

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))

def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`."""
    A, B = jax.jacfwd(f, argnums=(0, 1))(s, u)
    c = f(s, u) - A@s - B@u
    return A, B, c

def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""
    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return integrator

def scp_iteration(f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, ρ):
    """Solve a single SCP sub-problem for ReachBot trajectory generation

    Parameters:
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    s_prev : numpy.ndarray
        The state trajectory around which the problem is convexified (2-D).
    u_prev : numpy.ndarray
        The control trajectory around which the problem is convexified (2-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    ρ : float
        Trust region radius.

    Returns:
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : float
        The SCP sub-problem cost.
    """
    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    s_cvx = cp.Variable((N + 1, n))
    u_cvx = cp.Variable((N, m))

    # Construct the convex SCP sub-problem.
    objective = 0.0
    constraints = []
    
    for k in range(N): # Summation of quadratic costs (k = 0 to N-1)
        objective += cp.quad_form(s_cvx[k] - s_goal, Q) # State Cost
        objective += cp.quad_form(u_cvx[k], R) # Control Cost
    objective += cp.quad_form(s_cvx[N] - s_goal, P) # Terminal State Cost (proxy for terminal constraint)
    # DEFINE CONSTRAINTS
    constraints.append(s_cvx[0] == s0)  # Initial state constraint
    for k in range(N): # Constraints on state dynamics, control input, and trust region
        constraints.append(s_cvx[k + 1] == A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k]) # linearized state dynamics
        # constraints.append(cp.norm(u_cvx[k], 'inf') <= u_max) # control input constraint
        constraints.append(cp.norm(u_cvx[k][:2], 'inf') <= u_max)  # Torque constraints
        # 0.02 = 66 iterations, minimumm
        constraints.append(cp.abs(u_cvx[k][2]) <= 0.5)                    # Force constraint
        constraints.append(cp.norm(s_cvx[k] - s_prev[k], 'inf') <= ρ) # trust region on s_k
        constraints.append(cp.norm(u_cvx[k] - u_prev[k], 'inf') <= ρ) # trust region on u_k
    constraints.append(cp.norm(s_cvx[N] - s_prev[N], 'inf') <= ρ) # final state trust region
    
    # Solve CVXPY optimization problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    # prob.solve(solver=cp.SCS, verbose=True, max_iters=50000)
    prob.solve()
    if prob.status != "optimal":
        raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J

def solve_traj_scp(f, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters):
    """Solve the ReachBot optimal trajectory generation problem with SCP.

    Arguments:
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    ρ : float
        Trust region radius.
    eps : float
        Termination threshold for SCP.
    max_iters : int
        Maximum number of SCP iterations.

    Returns:
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : numpy.ndarray
        A 1-D array where `J[i]` is the SCP sub-problem cost after the i-th
        iteration, for `i = 0, 1, ..., (iteration when convergence occured)`
    """
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize dynamically feasible nominal trajectories
    u = np.zeros((N, m)) # zero control effort
    s = np.zeros((N + 1, n))
    s[0] = s0
    for k in range(N):
        s[k + 1] = f(s[k], u[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, N, P, Q, R, u_max, ρ)
        dJ = np.abs(J[i + 1] - J[i])
        prog_bar.set_postfix({"objective change": "{:.5f}".format(dJ)})
        if dJ < eps:
            converged = True
            print("SCP converged after {} iterations.".format(i))
            break
    if not converged:
        raise RuntimeError("SCP did not converge!")
    J = J[1 : i + 1]
    return s, u, J

# --------------------------------------------------------------------------------
# TRAJECTORY PARAMETERS
# Initial State
# x0, y0, z0 = 0, 0, 0.15
x0, y0, z0 = 0, 0, 0.05
θ1_init, θ2_init, d3_init = inv_kinematics(x0, y0, z0)
# Goal State
# xg, yg, zg = -0.015, -0.115, 0.95
# xg, yg, zg = -0.01, -0.07, 0.95 # for gripper attached (kinda works)
xg, yg, zg = 0.5, 0.5, 1.5
θ1_goal, θ2_goal, d3_goal = inv_kinematics(xg, yg, zg)
# --------------------------------------------------------------------------------

# SETUP
n = 6 # state dimension
m = 3 # control dimension

print(θ1_init, θ2_init, d3_init)
print(θ1_goal, θ2_goal, d3_goal)

s0 = np.array([θ1_init, θ2_init, d3_init, 0, 0, 0]) # initial state (trajectory start)
s_goal = np.array([θ1_goal, θ2_goal, d3_goal, 0, 0, 0]) # goal state (trajectory end)

dt = 0.1  # discrete time resolution
T = 10.0  # total simulation time

P = 1e3 * np.eye(n)  # terminal state cost matrix
Q = np.diag([0.01, 0.01, 0.001, 0.001, 0.001, 0.0001])*2  # state cost matrix
R = np.diag([0.1, 0.1, 0.001])  # control cost matrix

ρ = 1  # trust region parameter
u_max = 1.5  # control effort bound (motor torque limit) [N/m]
# eps = 5e-1  # convergence tolerance
eps = 1e-6
max_iters = 100  # maximum number of SCP iterations

# Initialize the discrete-time manipulator dynamics
fd = jax.jit(discretize(boom_dynamics, dt))

# Solve the swing-up problem with SCP
t = np.arange(0.0, T + dt, dt)
N = t.size - 1
s1, u1, J = solve_traj_scp(fd, s0, s_goal, N, P, Q, R, 0.025, ρ, eps, max_iters)
s2, u2, J = solve_traj_scp(fd, s0, s_goal, N, P, Q, R, 0.075, ρ, eps, max_iters)
s3, u3, J = solve_traj_scp(fd, s0, s_goal, N, P, Q, R, 0.25, ρ, eps, max_iters)

# Simulate open-loop control
for k in range(N):
    s1[k + 1] = fd(s1[k], u1[k])
    s2[k + 1] = fd(s2[k], u2[k])
    s3[k + 1] = fd(s3[k], u3[k])

pos1 = np.zeros((N, 3))
pos2 = np.zeros((N, 3))
pos3 = np.zeros((N, 3))
for i in range(N):
    pos1[i] = fwd_kinematics(s1[i])
    pos2[i] = fwd_kinematics(s2[i])
    pos3[i] = fwd_kinematics(s3[i])


## WRITE TO SAVEFILE
    # np.save('./DATA/strict_traj.npy', s1)
    # np.save('./DATA/moderate_traj.npy', s2)
    # np.save('./DATA/lenient_traj.npy', s3)

PD_s, PD_u, PD_pos = generate_PD_trajectories()
## GENERATE PLOTS
plot_2x_trajectories(PD_pos, pos2)
plot_2x_control_efforts(PD_u, u2)
plot_2x_joint_traj(PD_s, s2, θ1_goal, θ2_goal, d3_goal)

# # animate_trajectory(pos)
# plt.figure(figsize=(10, 6))
# plot_control_efforts(u1, 'u1')
# # plot_control_efforts(u2, 'u2')
# plot_control_efforts(u3, 'u3')

# plt.xlabel('Time')
# plt.ylabel('Control Effort')
# plt.title('Control Efforts Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# plot_joint_traj(s, θ1_goal, θ2_goal, d3_goal)
