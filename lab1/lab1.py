import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import time
import sounddevice as sd

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

# Problem B.4
t = np.linspace(-1, 1, 1000)
n3_t = n(t + 1/4)
plot(n3_t, t, figsize=(8.0, 4.0), title='n3(t) = n(t+1/4)', plotLabel='n3(t) = n(t + 1/4)', xLabel='t', yLabel='n3(t)')
t = t + 1/4
n4_t = n(1/2*(t))
plot(n4_t, t, newGraph=False, figsize=(8.0, 4.0), title='n3(t) = n(t+1/4),n4(t) = n3(1/2 t)', plotLabel='n4(t) = n3(1/2 t)', xLabel='t', yLabel='n4(t),n3(t)')

# Problem C.1
t = np.linspace(-2,2, 1000)
f_t = np.exp(-2 * t) * np.cos(4 * np.pi * t)
u_t = np.heaviside(t, 1)
g_t = u_t * f_t
plot(g_t,t, title='C.1 (f(t) = e^-2tcos(4PI*T))', plotLabel= "g_t = u_t * f_t", xLabel='t', yLabel='g(t)')

# Problem C.2
t = np.linspace(-2,4,600) #-2:0.1:4
t = t + 1
f_t = np.exp(-2 * t) * np.cos(4 * np.pi * t)
u_t = np.heaviside(t, 1)
g_t = u_t * f_t
plot(g_t,t, title='C.2', plotLabel= "g_t = u_t * f_t", xLabel='t', yLabel='g(t)')

# Problem C.3
t = np.linspace(0,4,400) #0:0.01:4
u_t = np.heaviside(t, 1)
alpha = [1,3,5,7]

newFig = True
for i in range(0,4):
    if(i > 0):
        newFig = False
    f_t = np.exp(-2) * np.exp(-1 * alpha[i] * t) * np.cos(4 * np.pi * t)
    g_t = f_t * u_t
    plotLable = "alpha: " + str(alpha[i])
    plot(g_t,t, newGraph=newFig,title='C.3', plotLabel= plotLable , xLabel='t', yLabel='g(t)')

# Problem C.4

# D.1
# Load the MATLAB data file
data = sci.loadmat('lab1\ELE532_Lab1_Data.mat')

# Access the loaded data arrays
A = data['A']
B = data['B']
array_x_audio = data['x_audio']

# (a) Convert the matrix A into a column vector
a_result = A.flatten()
print(A)

# (b) Extract specific elements from A based on their indices
b_indices = np.array([1, 3, 6])  # 0-based indices
b_result = A.flatten()[b_indices]

# (c) Create a Boolean mask for elements in A greater than or equal to 0.2
c_mask = A >= 0.2

# (d) Extract elements from A based on the Boolean mask
d_result = A[c_mask]

# (e) Set elements in A that satisfy the condition to 0
A[c_mask] = 0

print("(a) Flattened A:")
print(a_result)
print("\n(b) Extracted elements from A based on indices:")
print(b_result)
print("\n(c) Boolean mask for elements >= 0.2:")
print(c_mask)
print("\n(d) Extracted elements from A where A >= 0.2:")
print(d_result)
print("\n(e) Matrix A after setting elements >= 0.2 to 0:")
print(A)

#D.2

# Get the dimensions of the matrix
rows, cols = B.shape

#a)
# Iterate through the matrix elements and set values below 0.01 to zero
start = time.time()
for i in range(rows):
    for j in range(cols):
        if abs(B[i, j]) < 0.01:
            B[i, j] = 0
end = time.time()
nestedTime = end-start

#b)
start = time.time()
B[B < 0.01] = 0
end = time.time()
pythonTime = end-start

print("\nnested time: " + str(nestedTime) + "\npython time:" + str(pythonTime))

#D.3

# Copy the data to a working array
working_array = np.copy(array_x_audio)

# Define the threshold for compression (adjust as needed)
threshold = 0.01

# Initialize a counter for zero-valued samples
zero_samples_count = 0

# Iterate through the working array and apply compression
for i in range(len(working_array)):
    if abs(working_array[i]) < threshold:
        working_array[i] = 0
        zero_samples_count += 1

# Print the number of zero-valued samples
print(f"\nNumber of zero-valued samples: {zero_samples_count}")

# Play the original audio (unprocessed) for comparison
sd.play(array_x_audio, 8000)
sd.wait()

# Play the processed audio with compression
sd.play(working_array, 8000)
sd.wait()

plt.show() #end of code
