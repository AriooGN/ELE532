import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from scipy.signal import lti, step
import sympy as sp

def plot(f_t, t, title=None, plotLabel=None, xLabel=None, yLabel=None, newGraph=True, figsize=(8.0, 4.0)):
    """
    Plot a function with optional labels and titles.

    Parameters:
        f_t (array-like): The function values to plot.
        t (array-like): The time or x-axis values.
        title (str, optional): The title of the plot.
        plotLabel (str, optional): The label for the plot.
        xLabel (str, optional): The label for the x-axis.
        yLabel (str, optional): The label for the y-axis.
        newGraph (bool, optional): Whether to create a new figure. Default is True.
        figsize (tuple, optional): The figure size. Default is (8.0, 4.0).
    """
    if newGraph:
        plt.figure(figsize=figsize)

    plt.plot(t, f_t, label=plotLabel)

    if title:
        plt.title(title)
    if xLabel:
        plt.xlabel(xLabel)
    if yLabel:
        plt.ylabel(yLabel)

    plt.grid(True)

    if plotLabel:
        plt.legend()

    plt.tight_layout()

# Part A (A.1, A.2, A.3)


# A.1: Determine characteristic roots of op-amp circuit
# Set component values
R = np.array([1e4, 1e4, 1e4])
C = np.array([1e-6, 1e-6])
C2 = np.array([1e-9, 1e-6])

# Determine coefficients for characteristic equation
A = [1, (1/R[0] + 1/R[1] + 1/R[2]) / C[1], 1 / (R[0] * R[1] * C[0] * C[1])]
A2 = [1, (1/R[0] + 1/R[1] + 1/R[2]) / C2[1], 1 / (R[0] * R[1] * C2[0] * C2[1])]

# Determine characteristic roots
lambda_values = np.roots(A)
lambda_values2 = np.roots(A2)

# Calculate polynomial with roots as lambda_values
p = np.poly([lambda_values[0], lambda_values[1]])

# A.2: Impulse Response
# Define symbolic variables
t = sp.Symbol('t', real=True)
y = sp.Function('y')(t)

# Define the differential equation and initial conditions
eqn = sp.diff(y, t, 2) + 300*sp.diff(y, t) + 10000*y
cond = {y.subs(t, 0): 0, sp.diff(y, t).subs(t, 0): 1}

# Solve the differential equation
sol = sp.dsolve(eqn, ics=cond)
h = -1/(R[0] * R[2] * C[0] * C[1]) * sol.rhs

# Generate time and h(t) values for plotting
time_values = np.linspace(0, 0.1, 200)
h_values = [float(h.subs(t, val)) for val in time_values]

# Use the plot function to plot the impulse response h(t) with the calculated coefficients A and B
plot(h_values, time_values, title='A.2', xLabel='Time [s]', yLabel='h(t)')

# Problem A.3   

def CH2MP2(R, C):
    # Determine the coefficients for the characteristic equation
    A = [1, (1/R[0] + 1/R[1] + 1/R[2]) / C[1], 1 / (R[0] * R[1] * C[0] * C[1])]
    
    # Determine characteristic roots
    roots = np.roots(A)
    
    return roots

lambda_ = CH2MP2([1e4, 1e4, 1e4],[1e-9, 1e-6])
print("A.3 Lambda: ", lambda_)
# Part B (B.1, B.2, B.3)

# Problem B.1
"""
script_path = 'C:\\Users\\arian\\OneDrive\\Documents\\SimRacing\\ELE532\\lab2\\CH2MP4.m'
eng = matlab.engine.start_matlab()
eng.eval(f"run('{script_path}')", nargout=0)
input()
eng.quit()
print("eng.quit")
"""
# Problem B.2

# Define the functions x(t) and h(t)
x = lambda t: np.heaviside(t , 1) - np.heaviside(t - 2, 1)
h = lambda t: (t+1) * (np.heaviside(t + 1, 1) - np.heaviside(t, 1))

# Define the time vector
t = np.arange(-2, 5, 0.01)

x_t = x(t)
h_t = h(t)

y_t = np.convolve(x_t, h_t, 'same') * 0.01

plt.figure(figsize=(6, 12))

# Subplot for x(t)
plt.subplot(3,1,1)
plt.plot(t, x(t), label='x(t)')
plt.title('x(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for h(t)
plt.subplot(3,1,2)
plt.plot(t, h(t), label='h(t)')
plt.title('h(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for y(t)
plt.subplot(3, 1, 3)
plt.plot(t, y_t)
plt.title('y(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplots_adjust(hspace=0.5)

# Problem B.3

    #part a)
x1 = lambda t: (np.heaviside(t - 4, 0.5) - np.heaviside(t - 6, 0.5))
x2 = lambda t: (np.heaviside(t + 5, 0.5) - np.heaviside(t + 4, 0.5))
t = np.linspace(-10, 10, 1000)

x1_t = x1(t)
x2_t = x2(t)

convolution = np.convolve(x1_t, x2_t, 'same') * (t[1]-t[0])  # Multiply by dt for integration

plt.figure(figsize=(6, 12))


# Subplot for x1(t)
plt.subplot(3, 1, 1)
plt.plot(t, x1_t)
plt.title('x1(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for x2(t)
plt.subplot(3, 1, 2)
plt.plot(t, x2_t)
plt.title('x2(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for convolution
plt.subplot(3, 1, 3)
plt.plot(t, convolution)
plt.title('Convolution of x1(t) and x2(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplots_adjust(hspace=0.5)

    #part b)
x1 = lambda t: (np.heaviside(t - 3, 0.5) - np.heaviside(t - 5, 0.5))
x2 = lambda t: (np.heaviside(t + 5, 0.5) - np.heaviside(t + 3, 0.5))

t = np.linspace(-10, 10, 1000)

x1_t = x1(t)
x2_t = x2(t)

convolution = np.convolve(x1_t, x2_t, 'same') * (t[1]-t[0])  # Multiply by dt for integration

plt.figure(figsize=(6, 12))

# Subplot for x1(t)
plt.subplot(3, 1, 1)
plt.plot(t, x1_t)
plt.title('x1(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for x2(t)
plt.subplot(3, 1, 2)
plt.plot(t, x2_t)
plt.title('x2(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for convolution
plt.subplot(3, 1, 3)
plt.plot(t, convolution)
plt.title('Convolution of x1(t) and x2(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplots_adjust(hspace=0.5)

    #part h)
x1 = lambda t: np.exp(t) * (np.heaviside(t + 2, 0.5) - np.heaviside(t, 0.5))
x2 = lambda t: np.exp(-2 * t) * (np.heaviside(t, 0.5) - np.heaviside(t - 1, 0.5))

t = np.linspace(-4, 3, 1000)  # Extended range to capture the essence of both functions

x1_t = x1(t)
x2_t = x2(t)

convolution = np.convolve(x1_t, x2_t, 'same') * (t[1]-t[0])  # Multiply by dt for integration

plt.figure(figsize=(6, 12))

# Subplot for x1(t)
plt.subplot(3, 1, 1)
plt.plot(t, x1_t)
plt.title('x1(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for x2(t)
plt.subplot(3, 1, 2)
plt.plot(t, x2_t)
plt.title('x2(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot for convolution
plt.subplot(3, 1, 3)
plt.plot(t, convolution)
plt.title('Convolution of x1(t) and x2(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplots_adjust(hspace=0.5)

# Part C (C.1,C.2,C.3)

# Define the time range
t = np.arange(-1, 5, 0.001)

# Define the impulse response functions h(t) for each system
h1_t = np.exp(t) * np.heaviside(t, 0)
h2_t = 4 * np.exp(-t) * np.heaviside(t, 0)
h3_t = 4 * np.exp(-t) * np.heaviside(t, 0)
h4_t = 4 * (np.exp(-t / 5) - np.exp(-t)) * np.heaviside(t, 0)

# Plot the impulse response functions
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(t, h1_t)
plt.title('System S1: Impulse Response h1(t)')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, h2_t)
plt.title('System S2: Impulse Response h2(t)')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, h3_t)
plt.title('System S3: Impulse Response h3(t)')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, h4_t)
plt.title('System S4: Impulse Response h4(t)')
plt.grid(True)

plt.subplots_adjust(hspace=0.5)

#part c
script_path = 'C:\\Users\\arian\\OneDrive\\Documents\\SimRacing\\ELE532\\lab2\\Partc.m'
eng = matlab.engine.start_matlab()
eng.eval(f"run('{script_path}')", nargout=0)
input()
eng.quit()
print("eng.quit")



plt.show()