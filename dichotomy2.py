import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Define the function
def f(x):
    return 4.0 - np.exp(x) - 2.0 * x ** 2.0


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
a, b = -2.0, 0.0
epsilon = 1e-5

# Run the bisection method
root, points = bisection_method(a, b, epsilon)

# Prepare data for plotting
x = np.linspace(a, b, 400)
y = f(x)

# Plot setup
fig, ax = plt.subplots()
ax.plot(x, y, label='f(x)')
aline = ax.axvline(x=a, color='red')
bline = ax.axvline(x=b, color='red')
ax.axhline(0, color='black', lw=1)
root_line = ax.axvline(x=root, color='green', linestyle='--', label='Approximated Root')
ax.legend(loc='upper right')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Bisection Method Animation')
ax.set_ylim(-5, 5)
ax.set_ylim(-5, 5)

fill_between = None
text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top')

# Update function for animation
def update(frame):
    global fill_between
    a, b, c = points[frame]
    aline.set_xdata([a, a])
    bline.set_xdata([b, b])

    # Clear the previous fill
    if fill_between:
        fill_between.remove()

    text.set_text(f'Current approximated root: {c:.5f}')

    # Fill between new a and b
    fill_between = ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], a, b, color='red', alpha=0.3)
    return aline, bline, fill_between, text


# Create animation
ani = FuncAnimation(fig, update, frames=len(points), blit=True, repeat=True)
ani.save('animations/dichotomy2.gif', writer='imagegick', dpi=300)


# Show animation
plt.show()
