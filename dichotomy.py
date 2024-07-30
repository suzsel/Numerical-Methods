import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x):
    return 4.0 - np.exp(x) - 2.0 * x**2.0

# Bisection method implementation
def bisection_method(a, b, epsilon=1e-5):
    points = []
    while b - a > epsilon:
        c = (a + b) / 2.0
        points.append((a, b, c))
        if f(b) * f(c) < 0:
            a = c
        else:
            b = c
    return (a + b) / 2.0, points

# Initialize the range and epsilon
a, b = 0.0, 2.0
epsilon = 1e-5

# Run the bisection method
root, points = bisection_method(a, b, epsilon)

# Prepare data for plotting
x = np.linspace(a, b, 400)
y = f(x)

# Plot setup
fig, ax = plt.subplots()
ax.plot(x, y, label='f(x)')
line, = ax.plot([], [], 'ro-', lw=2)
ax.axhline(0, color='black', lw=1)
root_line = ax.axvline(x=root, color='green', linestyle='--', label='Approximated Root')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Bisection Method Animation')

# Update function for animation
def update(frame):
    a, b, c = points[frame]
    line.set_data([a, b], [f(a), f(b)])
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=len(points), blit=True, repeat=True)
ani.save('animations/dichotomy1.gif', writer='imagegick', dpi=300)

# Show animation
plt.show()
