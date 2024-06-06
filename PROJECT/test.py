from functools import partial
import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from boom_dynamics import boom_dynamics, fwd_kinematics, inv_kinematics

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`."""
    A, B = jax.jacfwd(f, argnums=(0, 1))(s, u)
    c = f(s, u) - A @ s - B @ u
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
    """Solve a single SCP sub-problem for ReachBot trajectory generation."""
    u_max_torque = 1.5
    u_max_force = 15

    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    s_cvx = cp.Variable((N + 1, n))
    u_cvx = cp.Variable((N, m))

    # Construct the convex SCP sub-problem.
    objective = 0.0
    constraints = []

    for k in range(N):
        objective += cp.quad_form(s_cvx[k] - s_goal, Q)
        objective += cp.quad_form(u_cvx[k], R)
    objective += cp.quad_form(s_cvx[N] - s_goal, P)

    constraints.append(s_cvx[0] == s0)
    for k in range(N):
        constraints.append(s_cvx[k + 1] == A[k] @ s_cvx[k] + B[k] @ u_cvx[k] + c[k])
        constraints.append(cp.norm(u_cvx[k][:2], 'inf') <= u_max_torque)  # Torque constraints
        constraints.append(u_cvx[k][2] <= u_max_force)                    # Force constraint
        constraints.append(cp.norm(s_cvx[k] - s_prev[k], 'inf') <= ρ)
        constraints.append(cp.norm(u_cvx[k] - u_prev[k], 'inf') <= ρ)
    constraints.append(cp.norm(s_cvx[N] - s_prev[N], 'inf') <= ρ)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    # Try solving with OSQP first
    try:
        prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise cp.error.SolverError(f"Solver status: {prob.status}")
    except cp.error.SolverError as e:
        print(f"OSQP solver failed: {e}. Trying SCS solver...")
        prob.solve(solver=cp.SCS, verbose=False, max_iters=25000)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"SCP solve failed. Problem status: {prob.status}")

    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J

def solve_traj_scp(f, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters):
    """Solve the ReachBot optimal trajectory generation problem with SCP."""
    n = Q.shape[0]
    m = R.shape[0]

    u = np.zeros((N, m))
    s = np.zeros((N + 1, n))
    s[0] = s0
    for k in range(N):
        s[k + 1] = f(s[k], u[k])

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
x0, y0, z0 = 0, 0, 1
θ1_init, θ2_init, d3_init = inv_kinematics(x0, y0, z0)
xg, yg, zg = 0.2, 0.2, 1.2
θ1_goal, θ2_goal, d3_goal = inv_kinematics(xg, yg, zg)
# --------------------------------------------------------------------------------

# SETUP
n = 6
m = 3

print(θ1_init, θ2_init, d3_init)
print(θ1_goal, θ2_goal, d3_goal)

s0 = np.array([θ1_init, θ2_init, d3_init, 0, 0, 0])
s_goal = np.array([θ1_goal, θ2_goal, d3_goal, 0, 0, 0])

dt = 0.1
T = 10.0

P = 1e3 * np.eye(n)
Q = np.diag([1e-1, 1e-1, 1e-2, 1e-3, 1e-3, 1e-4])
R = np.diag([1e-1, 1e-1, 1e-2])

ρ = 1.0
u_max = 0.28
eps = 5e-1
max_iters = 100

fd = jax.jit(discretize(boom_dynamics, dt))

t = np.arange(0.0, T + dt, dt)
N = t.size - 1  # Ensure N is defined before using it in the function call

s, u, J = solve_traj_scp(fd, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters)

for k in range(N):
    s[k + 1] = fd(s[k], u[k])
