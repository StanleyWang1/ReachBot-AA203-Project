import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def plot_2x_trajectories(data1, data2):
    """
    Plots two trajectories in 3D space with a point at the endpoints.

    Parameters:
    - data1: np.ndarray of shape (N, 3), the first trajectory data.
    - data2: np.ndarray of shape (N, 3), the second trajectory data.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first trajectory
    ax.scatter(data1[0, 0], data1[0, 1], data1[0, 2], color='cyan', s=100, label='Initial Position')
    ax.scatter(data1[-1, 0], data1[-1, 1], data1[-1, 2], color='orange', s=100, label='Goal Position')

    ax.plot(data1[:, 0], data1[:, 1], data1[:, 2], label='PD Controller', color='red')
    ax.plot(data2[:, 0], data2[:, 1], data2[:, 2], label='SCP Optimization (u < 0.075)', color='blue')

    # Set labels and title
    ax.set_title('3D Trajectories')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Set equal axes
    ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

    plt.show()

def plot_2x_control_efforts(u1, u2):
    """
    Plots the control efforts for 3 joints over time for three different control trajectories.
    
    Parameters:
    - u1, u2, u3: np.ndarray of shape (N, 3), the control efforts for 3 joints over time.
    """
    time = np.arange(u1.shape[0]) * 0.1  # Generate a time array
    
    # Define line styles for each trajectory
    line_styles = ['-', '--', ':']
    
    # Define colors for each joint component
    colors = ['purple', 'orange', 'teal']
    
    # Plot the control efforts for u1
    plt.plot(time, u1[:, 0], label='θ1 (PD Control)', color=colors[0], linestyle=line_styles[0])
    plt.plot(time, u1[:, 1], label='θ2 (PD Control)', color=colors[1], linestyle=line_styles[0])
    plt.plot(time, u1[:, 2], label='d3 (PD Control)', color=colors[2], linestyle=line_styles[0])
    
    # Plot the control efforts for u2
    plt.plot(time, u2[:, 0], label='θ1 (Optimal Control)', color=colors[0], linestyle=line_styles[1])
    plt.plot(time, u2[:, 1], label='θ2 (Optimal Control)', color=colors[1], linestyle=line_styles[1])
    plt.plot(time, u2[:, 2], label='d3 (Optimal Control)', color=colors[2], linestyle=line_styles[1])
    
    plt.xlabel('Time')
    plt.ylabel('Joint Control Effort')
    plt.title('Control Efforts')
    # plt.legend()
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.1))
    plt.grid(True)
    plt.show()

def plot_2x_joint_traj(s1, s2, θ1_goal, θ2_goal, d3_goal):
    time = np.arange(s1.shape[0]) * 0.1  # Generate a time array

    # fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig, axs = plt.subplots(3, 1)
    
    # Plot θ1 trajectories
    axs[0].plot(time, s1[:, 0], label='θ1 (PD Control)', color='red')
    axs[0].plot(time, s2[:, 0], label='θ1 (Optimal Control)', color='blue')
    axs[0].axhline(θ1_goal, color='gray', linestyle='--', linewidth=2, label='θ1 Goal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('θ1(t)')
    # axs[0].set_title('θ1 Joint Trajectories')
    axs[0].legend()
    axs[0].grid(True)

    # Plot θ2 trajectories
    axs[1].plot(time, s1[:, 1], label='θ2 (PD Control)', color='red')
    axs[1].plot(time, s2[:, 1], label='θ2 (Optimal Control)', color='blue')
    axs[1].axhline(θ2_goal, color='gray', linestyle='--', linewidth=2, label='θ2 Goal')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('θ2(t)')
    # axs[1].set_title('θ2 Joint Trajectories')
    axs[1].legend()
    axs[1].grid(True)

    # Plot d3 trajectories
    axs[2].plot(time, s1[:, 2], label='d3 (PD Control)', color='red')
    axs[2].plot(time, s2[:, 2], label='d3 (Optimal Control)', color='blue')
    axs[2].axhline(d3_goal, color='gray', linestyle='--', linewidth=2, label='d3 Goal')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('d3(t)')
    # axs[2].set_title('Joint Trajectories')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()