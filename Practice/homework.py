import numpy as np
import matplotlib.pyplot as plt

def plot(f_t, t, newGraph=True, plotLabel=''):
    plt.plot(t, f_t, label=plotLabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

t = np.linspace(-100,100,1000)
f_t = np.heaviside(t,1) * t*t

plot(f_t=f_t,t=t,plotLabel='u(t)')

plt.show()