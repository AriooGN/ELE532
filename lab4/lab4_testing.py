import numpy as np
import matplotlib.pyplot as plt

N = 100
PulseWidth = 10

# Generate the square wave signal

# Generate a time vector for plotting
t = np.arange(N)

# Plot using the 'step' function for a more square-like appearance
plt.step(t, x, antialiased=False)



plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Square Wave Signal')

plt.grid(True)
plt.show()
