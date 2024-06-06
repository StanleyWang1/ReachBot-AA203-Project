import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Define fwd_kinematics, inv_kinematics, and boom_dynamics functions
def fwd_kinematics(s):
    θ1, θ2, d3, _, _, _ = s
    c1, s1 = jnp.cos(θ1), jnp.sin(θ1)
    c2, s2 = jnp.cos(θ2), jnp.sin(θ2)

    x = 1/2 * d3 * (s2*(c1+1) - c2*(c1-1))
    y = -1/jnp.sqrt(2) * s1 * d3 * (c2 - s2)
    z = 1/2 * d3 * (c2*(c1+1) - s2*(c1-1))

    return x, y, z

def inv_kinematics(x, y, z):
    θ1 = jnp.atan2(-jnp.sqrt(2) * y, (-x+z))
    θ2 = -jnp.acos((x + z) / jnp.sqrt(2*(x**2+y**2+z**2))) + jnp.pi/4
    d3 = jnp.sqrt(x**2+y**2+z**2)
    return θ1, θ2, d3

def boom_dynamics(s, u):
    θ1, θ2, d3, dθ1, dθ2, dd3 = s
    c1, s1 = jnp.cos(θ1), jnp.sin(θ1)
    c2, s2 = jnp.cos(θ2), jnp.sin(θ2)

    x, y, z = fwd_kinematics(s)

    
    u_reshape = jnp.reshape(u, (3, 1))

    ds = jnp.array([dθ1, dθ2, dd3, u[0], u[1], u[2]])
    return ds

def generate_PD_trajectories():
    # PD controller parameters
    Kp = 5  # Proportional gain
    Kd = 5   # Derivative gain

    # Initial and goal positions
    x0, y0, z0 = 0, 0, 0.05
    xg, yg, zg = 0.5, 0.5, 1.5

    # Calculate initial joint angles
    theta0 = np.array(inv_kinematics(x0, y0, z0))

    # Calculate goal joint angles
    theta_goal = np.array(inv_kinematics(xg, yg, zg))

    # Define PD controller function
    def pd_controller(theta, theta_goal, theta_dot):
        return Kp * (theta_goal - theta) - Kd * theta_dot 

    # Simulation parameters
    dt = 0.1  # Time step
    T = 10.0   # Total simulation time
    timesteps = int(T / dt)

    # Initialize variables
    theta = theta0
    theta_dot = np.zeros_like(theta0)
    theta_ddot = np.zeros_like(theta0)
    state = np.concatenate([theta, theta_dot])

    state_vals = np.zeros((timesteps+1, 6))
    state_vals[0] = state

    # Store trajectories
    theta_trajectory = np.zeros((timesteps, len(theta)))
    theta_dot_trajectory = np.zeros((timesteps, len(theta_dot)))
    theta_ddot_trajectory = np.zeros((timesteps, len(theta_ddot)))

    u_vals = np.zeros((timesteps, 3))
    # Simulation loop
    for tk in tqdm(range(timesteps)):
        # Compute control input
        u = pd_controller(theta, theta_goal, theta_dot)
        u_vals[tk] = u

        # Update dynamics
        state_dot = boom_dynamics(state, u)
        state += state_dot * dt
        theta, theta_dot = state[:3], state[3:]
        
        state_vals[tk+1] = state
        
        # Store trajectories
        theta_trajectory[tk, :] = theta
        theta_dot_trajectory[tk, :] = theta_dot
        theta_ddot_trajectory[tk, :] = state_dot[3:]

    t = np.arange(0.0, T + dt, dt)
    N = t.size - 1

    pos = np.zeros((N, 3))
    for ti in range(N):
        pos[ti] = fwd_kinematics(np.array([theta_trajectory[ti, 0], theta_trajectory[ti, 1], theta_trajectory[ti, 2], 0, 0, 0]))
    return state_vals, u_vals, pos