import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import jax
import jax.numpy as jnp

def fwd_kinematics(s):
    """
    Compute the forward kinematics of the ReachBot boom manipulator
    Parameters:
        s - state vector of joint positions/velocities [θ1, θ2, d3, dθ1, dθ2, dd3]
    Returns:
        x, y, z - resulting end effector Cartesian position 
    """
    # Extract joint positions and construct trig terms
    θ1, θ2, d3, _, _, _ = s
    c1, s1 = jnp.cos(θ1), jnp.sin(θ1)
    c2, s2 = jnp.cos(θ2), jnp.sin(θ2)

    # Forward Kinematics
    x = 1/2 * d3 * (s2*(c1+1) - c2*(c1-1))
    y = -1/jnp.sqrt(2) * s1 * d3 * (c2 - s2)
    z = 1/2 * d3 * (c2*(c1+1) - s2*(c1-1))

    return x, y, z

def inv_kinematics(x, y, z):
    """
    Compute the forward kinematics of the ReachBot boom manipulator
    Parameters:
        x, y, z - end effector Cartesian position 
    Returns:
        θ1, θ2, d3 - joint positions yielding given position
    """
    θ1 = jnp.atan2(-jnp.sqrt(2) * y, (-x+z))
    θ2 = -jnp.acos((x + z) / jnp.sqrt(2*(x**2+y**2+z**2))) + jnp.pi/4
    d3 = jnp.sqrt(x**2+y**2+z**2)
    return θ1, θ2, d3

def boom_dynamics(s, u):
    """
    Compute the state derivative of the ReachBot boom manipulator
    Parameters:
        s - state vector of joint positions/velocities [θ1, θ2, d3, dθ1, dθ2, dd3]
        u - control input vector of joint torques [τ1, τ2, f3]
    Returns:
        ds - state derivative [dθ1, dθ2, dd3, ddθ1, ddθ2, ddd3]
    """

    # Physical Constants
    λb = 0.12 # linear mass density of boom [kg/m]
    mEE = 0.1042 # mass of end effector [kg]
    g = 9.81 # gravitational accel on EARTH [m/s^2]

    # Decompose state input to components (joint positions/velocities)
    θ1, θ2, d3, dθ1, dθ2, dd3 = s
    c1, s1 = jnp.cos(θ1), jnp.sin(θ1)
    c2, s2 = jnp.cos(θ2), jnp.sin(θ2)

    # Forward Kinematics
    x, y, z = fwd_kinematics(s)

    # Masses
    mb = λb * d3 # mass of tape boom [kg]
    mboom = mb + mEE # total mass of boom system (tape + EE)

    # Inertia Tensor of Boom (tape + EE)
    Iboom = (1/3 * mb + mEE) * jnp.array([[y**2 + z**2, -x*y, -x*z],
                                  [-x*y, x**2 + z**2, -y*z],
                                  [-x*z, -y*z, x**2 + y**2]])
    
    # Jacobians
    Jv = jnp.array([[-1/2*d3*(s2-c2)*s1, 1/2*d3*(c2*(c1+1) + s2*(c1-1)), 1/2*(s2*(c1+1) - c2*(c1-1))],
                    [-c1/jnp.sqrt(2)*d3*(c2-s2), -s1/jnp.sqrt(2)*d3*(-s2-c2), -1/jnp.sqrt(2)*s1*(c2 - s2)],
                    [-1/2*d3*(c2-s2)*s1, 1/2*d3*(-s2*(c1+1) - c2*(c1-1)), 1/2*(c2*(c1+1) - s2*(c1-1))]])
                    
    Jω = jnp.array([[1/2*(1-c1), s2*1/2*(1+c1) + c2*1/2*(1-c1), 0],
                    [-s1/jnp.sqrt(2), s2*s1/jnp.sqrt(2) - c2*s1/jnp.sqrt(2), 0],
                    [1/2*(1+c1), s2*1/2*(1-c1) + c2*1/2*(1+c1), 0]])
    
    # Mass Matrix
    M = mboom*Jv.T@Jv + Jω.T@Iboom@Jω

    # Gravity Vector
    G = 1/2 * (1/2*mb + mEE) * g * jnp.array([[d3 * (-c2*s1 + s2*s1)],
                                        [d3 * (-s2*(c1+1) - c2*(c1-1))],
                                        [c2*(c1+1) - s2*(c1-1)]])
    
    # Compute Accelerations
    u_reshape = jnp.reshape(u, (3, 1))
    ddq = jnp.linalg.inv(M) @ (u_reshape - G)
    ddq = jnp.squeeze(ddq) # remove extra dimension 

    # Return state derivative
    # ds = jnp.array(
    #     [
    #         dθ1,
    #         dθ2,
    #         dd3,
    #         ddq[0],
    #         ddq[1],
    #         ddq[2]
    #     ]
    # )
    # State derivative based on acceleration control
    ds = jnp.array(
        [
            dθ1,
            dθ2,
            dd3,
            u[0],
            u[1],
            u[2]
        ]
    )
    return ds

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

def test_boom_dynamics():
    # Define a sample state and input
    s0 = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    u = jnp.array([0.5, 0.0, 0.0])

    # print(fwd_kinematics(s0))
    # print(inv_kinematics(0.5, 0.5, 0))

    # # Call the dynamics function
    # ds = boom_dynamics(s, u)
    fd = jax.jit(discretize(boom_dynamics, 0.1))

    N = 100
    data = np.zeros((N, 6))
    data[0] = s0
    for i in range(1, N):
        data[i] = fd(data[i-1], u)

    pos = np.zeros((N, 3))
    for i in range(N):
        pos[i] = fwd_kinematics(data[i])
    return pos
    # # Print results for debugging
    # print("State derivative (ds):", ds)

    # # Check dimensions
    # assert ds.shape == (6,), "State derivative should have 6 components."

    # # Check values
    # print("ddθ1:", ds[3])
    # print("ddθ2:", ds[4])
    # print("dd3:", ds[5])

    # Further checks and debugging steps
    # Print intermediate values if needed, for example:
    # mb = λb * s[2]
    # print("mb (mass of tape boom):", mb)
    # This helps ensure each step is computed correctly

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

# Run the test
data = test_boom_dynamics()

animate_trajectory(data)