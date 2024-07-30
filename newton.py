import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

epsilon = 1e-5

def f(x):
    return np.exp(x) + x**5 - x**4 - 12

def df(x):
    return (f(x + epsilon) - f(x)) / epsilon

# Newton's method implementation
def newton_method(x0, tolerance = epsilon, max_iterations=100):
    x = x0
    for _ in range(max_iterations):
        x_new = x - f(x) / df(x)
        yield x, x_new
        if abs(x_new - x) < tolerance:
            break
        x = x_new

# Initial guess
x0 = 0.1

# Generate data for animation
data = list(newton_method(x0))

# Prepare figure
fig, ax = plt.subplots()
x = np.linspace(0, 2, 400)
y = f(x)
ax.plot(x, y, label='$f(x) = x^2 - 2$')
ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)
ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax.legend()
line, = ax.plot([], [], 'ro-', lw=2)

# Initialization function
def init():
    line.set_data([], [])
    return line,

# Animation update function
def update(frame):
    x0, x1 = frame
    line.set_data([x0, x1], [f(x0), 0])
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=data, init_func=init, blit=True, interval=100, repeat=False)
ani.save("newton_method_animation.gif", writer='pillow', dpi=300)

# Show plot
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton\'s Method')
plt.show()
