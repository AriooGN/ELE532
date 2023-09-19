import numpy as np
import matplotlib.pyplot as plt

# Function for plotting
def plot(f_t, t, newGraph=True, figsize=(8.0, 4.0), title='', plotLabel='', xLabel='', yLabel=''):
    if newGraph:
        plt.figure(figsize=figsize)
    plt.plot(t, f_t, label=plotLabel)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# Problem A.1
t = np.linspace(-2, 2, 8)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t, figsize=(8.0, 4.0), title='f(t) = e^(-t) * cos(2πt) plot', plotLabel='e^(-t) * cos(2πt)', xLabel='t', yLabel='f(t)')

t = np.linspace(-2, 2, 400)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t, newGraph=False, figsize=(8.0, 4.0), title='f(t) = e^(-t) * cos(2πt) plot', plotLabel='e^(-t) * cos(2πt)', xLabel='t', yLabel='f(t)')

# Problem A.2
t = np.linspace(-2, 2, 5)
f_t = np.exp(t)
plot(f_t, t, figsize=(8.0, 4.0), title='f(t) = e^(-t) plot', plotLabel='e^(-t)', xLabel='t', yLabel='f(t)')
plt.xticks(np.arange(-2, 2.1, 1))

# Problem A.3
t = np.linspace(-2, 2, 8)
f_t = np.exp(-t) * np.cos(2 * np.pi * t)
plot(f_t, t, figsize=(8.0, 4.0), title='f(t) = e^(-t) * cos(2πt) plot', plotLabel='e^(-t) * cos(2πt)', xLabel='t', yLabel='f(t)')

t = np.linspace(-2, 2, 5)
f_t = np.exp(t)
plot(f_t, t, newGraph=False, figsize=(8.0, 4.0), title='f(t) = e^(-t) plot', plotLabel='e^(-t)', xLabel='t', yLabel='f(t)')

# Problem B.1
t = np.linspace(-1, 2, 1000)
p_t = np.heaviside(t, 1) - np.heaviside(t - 1, 1)
plot(p_t, t, figsize=(8.0, 4.0), title='p(t) = u(t)−u(t −1) over (−1 ≤ t ≤ 2).', plotLabel='u(t)−u(t −1)', xLabel='t', yLabel='p(t)')

# Problem B.2
def r(t):
    return t * p_t

def n(t):
    return r(t) + r(-t + 2)

r_t = r(t)
n_t = n(t)

plot(r_t, t, figsize=(8.0, 4.0), title='r(t) = tp(t)', plotLabel='r(t) = tp(t)', xLabel='t', yLabel='r(t)')
plot(n_t, t, newGraph=False, figsize=(8, 4), title='r(t) = tp(t) & n(t) =r(t) + r(−t + 2).', plotLabel='n(t) = r(t) + r(−t + 2).', xLabel='t', yLabel='n(t)')

# Problem B.3
t = np.linspace(-1, 1, 1000)  # Adjust the time values for n1 and n2
n1_t = n(0.5 * t)
plot(n1_t, t, figsize=(8.0, 4.0), title='n1(t) = n(1/2 t)', plotLabel='n1(t) = n(1/2 t)', xLabel='t', yLabel='n1(t)')
t = t + (1/2)
n2_t = n(0.5 * (t))  # Adjust the time values for n2
# Plotting n1 and n2
plot(n2_t, t, newGraph=False, figsize=(8.0, 4.0), title='n2(t) = n1(t + 1/2)', plotLabel='n2(t) = n1(t + 1/2)', xLabel='t', yLabel='n2(t)')

plt.show()
