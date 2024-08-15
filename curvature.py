import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sympy as sp

def oscullating_plane(r_point, r_prime, r_double_prime):
    B = np.cross(r_prime, r_double_prime)

    plane_size = 0.2
    x = np.linspace(-plane_size, plane_size, 2)
    y = np.linspace(-plane_size, plane_size, 2)
    xx, yy = np.meshgrid(x, y)
    zz = -(B[0] * (xx) + B[1] * (yy)) / B[2]  # Plane equation B[0]*x + B[1]*y + B[2]*z = 0

    # Shift the plane to pass through the point r_point
    xx += r_point[0]
    yy += r_point[1]
    zz += r_point[2]

    return xx, yy, zz


# Define the parameter t symbolically
t = sp.symbols('t')

r_t = sp.Matrix([sp.exp(-t) * sp.cos(10*t), sp.exp(-t) * sp.sin(10*t), sp.exp(-t)])
# r_t = sp.Matrix([sp.cos(t), sp.sin(t), 2])

# 1st and 2nd derivatives
r_prime_t = r_t.diff(t)
r_double_prime_t = r_prime_t.diff(t)

# Convert symbolic expressions to numerical functions
r_func = sp.lambdify(t, r_t, modules='numpy')
r_prime_func = sp.lambdify(t, r_prime_t, modules='numpy')
r_double_prime_func = sp.lambdify(t, r_double_prime_t, modules='numpy')

# Define a range for t
t_values = np.linspace(0, 2 * np.pi, 4*100)

# Compute the vectors
r_values = np.array([r_func(t_val) for t_val in t_values])
r_prime_values = np.array([r_prime_func(t_val) for t_val in t_values])
r_double_prime_values = np.array([r_double_prime_func(t_val) for t_val in t_values])
print(r_values.shape)


# Set up the 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(10, 10)
ax.set_ylim(10, 10)
ax.set_zlim(10, 10)


# Function to update the plot
def update(num):
    ax.cla()  # Clear the current plot
    # Re-plot the original function
    ax.plot(r_values[:, 0], r_values[:, 1], r_values[:, 2], label='r(t)', color='blue')
    # Plot the derivatives
    ax.quiver(r_values[num, 0], r_values[num, 1], r_values[num, 2],
              r_prime_values[num, 0], r_prime_values[num, 1], r_prime_values[num, 2],
              color='red', length=0.5, normalize=True, label="r'(t)")

    ax.quiver(r_values[num, 0], r_values[num, 1], r_values[num, 2],
              r_double_prime_values[num, 0], r_double_prime_values[num, 1], r_double_prime_values[num, 2],
              color='green', length=0.3, normalize=True, label="r''(t)")

    # Re-plot oscullating plane
    xx, yy, zz = oscullating_plane(r_values[num].reshape(-1), r_prime_values[num].reshape(-1), r_double_prime_values[num].reshape(-1))
    ax.plot_surface(xx, yy, zz, color='purple', alpha=0.5,
                    label='Osculating Plane')

    # Set labels and legend again
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector-Valued Function and its Derivatives')
    ax.legend()


ani = FuncAnimation(fig, update, frames=len(t_values), interval=100, blit=False)

plt.show()
