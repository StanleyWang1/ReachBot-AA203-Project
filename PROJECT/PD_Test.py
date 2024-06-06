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
    λb = 0.12 # linear mass density of boom [kg/m]
    mEE = 0.1042 # mass of end effector [kg]
    g = 9.81 # gravitational accel on EARTH [m/s^2]

    θ1, θ2, d3, dθ1, dθ2, dd3 = s
    c1, s1 = jnp.cos(θ1), jnp.sin(θ1)
    c2, s2 = jnp.cos(θ2), jnp.sin(θ2)

    x, y, z = fwd_kinematics(s)

    mb = λb * d3 # mass of tape boom [kg]
    mboom = mb + mEE # total mass of boom system (tape + EE)

    Iboom = (1/3 * mb + mEE) * jnp.array([[y**2 + z**2, -x*y, -x*z],
                                          [-x*y, x**2 + z**2, -y*z],
                                          [-x*z, -y*z, x**2 + y**2]])
    
    Jv = jnp.array([[-1/2*d3*(s2-c2)*s1, 1/2*d3*(c2*(c1+1) + s2*(c1-1)), 1/2*(s2*(c1+1) - c2*(c1-1))],
                    [-c1/jnp.sqrt(2)*d3*(c2-s2), -s1/jnp.sqrt(2)*d3*(-s2-c2), -1/jnp.sqrt(2)*s1*(c2 - s2)],
                    [-1/2*d3*(c2-s2)*s1, 1/2*d3*(-s2*(c1+1) - c2*(c1-1)), 1/2*(c2*(c1+1) - s2*(c1-1))]])
                    
    Jω = jnp.array([[1/2*(1-c1), s2*1/2*(1+c1) + c2*1/2*(1-c1), 0],
                    [-s1/jnp.sqrt(2), s2*s1/jnp.sqrt(2) - c2*s1/jnp.sqrt(2), 0],
                    [1/2*(1+c1), s2*1/2*(1-c1) + c2*1/2*(1+c1), 0]])
    
    M = mboom*Jv.T@Jv + Jω.T@Iboom@Jω

    G = 1/2 * (1/2*mb + mEE) * g * jnp.array([[d3 * (-c2*s1 + s2*s1)],
                                              [d3 * (-s2*(c1+1) - c2*(c1-1))],
                                              [c2*(c1+1) - s2*(c1-1)]])
    
    u_reshape = jnp.reshape(u, (3, 1))
    ddq = jnp.linalg.inv(M) @ (u_reshape - G)
    ddq = jnp.squeeze(ddq)

    ds = jnp.array([dθ1, dθ2, dd3, u[0], u[1], u[2]])
    return ds

def animate_trajectory(data, frame_rate=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.plot(data[:, 0], data[:, 1], data[:, 2], 'gray', alpha=0.5)
    line, = ax.plot([], [], [], 'r', linewidth=2)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(num):
        line.set_data([0, data[num, 0]], [0, data[num, 1]])
        line.set_3d_properties([0, data[num, 2]])
        return line,

    ani = FuncAnimation(fig, update, frames=data.shape[0], init_func=init, blit=True, interval=frame_rate)
    plt.show()
    return ani

def plot_control_efforts(u):
    """
    Plots the control efforts for 3 joints over time.

    Parameters:
    - u: np.ndarray of shape (N, 3), the control efforts for 3 joints over time.
    """
    time = np.arange(u.shape[0])  # Generate a time array

    plt.figure(figsize=(10, 6))
    
    plt.plot(time, u[:, 0], label='Joint 1')
    plt.plot(time, u[:, 1], label='Joint 2')
    plt.plot(time, u[:, 2], label='Joint 3')
    
    plt.xlabel('Time')
    plt.ylabel('Control Effort')
    plt.title('Control Efforts Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

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
dt = 0.05  # Time step
T = 10.0   # Total simulation time
timesteps = int(T / dt)

# Initialize variables
theta = theta0
theta_dot = np.zeros_like(theta0)
theta_ddot = np.zeros_like(theta0)
state = np.concatenate([theta, theta_dot])

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
    
    # Store trajectories
    theta_trajectory[tk, :] = theta
    theta_dot_trajectory[tk, :] = theta_dot
    theta_ddot_trajectory[tk, :] = state_dot[3:]

t = np.arange(0.0, T + dt, dt)
N = t.size - 1

pos = np.zeros((N, 3))
for ti in range(N):
    pos[ti] = inv_kinematics(theta_trajectory[ti, 0], theta_trajectory[ti, 1], theta_trajectory[ti, 2])
animate_trajectory(pos)
plot_control_efforts(u_vals)

# # Plot the results
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(np.linspace(0, T, timesteps), theta_trajectory)
# plt.title('Joint Angles')
# plt.xlabel('Time [s]')
# plt.ylabel('Theta')

# plt.subplot(3, 1, 2)
# plt.plot(np.linspace(0, T, timesteps), theta_dot_trajectory)
# plt.title('Joint Velocities')
# plt.xlabel('Time [s]')
# plt.ylabel('Theta dot')

# plt.subplot(3, 1, 3)
# plt.plot(np.linspace(0, T, timesteps), theta_ddot_trajectory)
# plt.title('Joint Accelerations')
# plt.xlabel('Time [s]')
# plt.ylabel('Theta double dot')

# plt.tight_layout()
# plt.show()
