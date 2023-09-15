import numpy as np
import matplotlib.pyplot as plt

#function that will handle plotting
def plot(f_t, t, newGraph=True, figsize=(8.0, 4.0), title='', plotLabel='', xLabel='', yLabel=''):
    if newGraph:
        plt.figure(figsize=figsize)  # Create a new figure for each plot if param is untouched
    plt.plot(t, f_t, label=plotLabel)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid()
    plt.legend()
    plt.tight_layout()

# Problem A.1
t = np.linspace(-2, 2, 8)  # defining 1000 points between -2 and 2
f_t = np.exp(-t) * np.cos(2 * np.pi * t)  # defining function
plot(f_t, t, figsize=(8.0, 4.0), title='f(t) = e^(-t) * cos(2πt) plot', plotLabel='e^(-t) * cos(2πt)', xLabel='t', yLabel='f(t)')

t = np.linspace(-2, 2, 400)  # -2:0.1:2
f_t = np.exp(-t) * np.cos(2 * np.pi * t)  # defining function
plot(f_t, t, figsize=(8.0, 4.0), title='f(t) = e^(-t) * cos(2πt) plot', plotLabel='e^(-t) * cos(2πt)', xLabel='t', yLabel='f(t)')

# Problem A.2
t = np.linspace(-2, 2, 5)  # 5 points from -2 to 2
f_t = np.exp(t) #e^t
plot(f_t, t, figsize=(8.0, 4.0), title='f(t) = e^(-t) plot', plotLabel='e^(-t)', xLabel='t', yLabel='f(t)')
plt.xticks(np.arange(-2, 2.1, 1))  # adjust the ticks so they also go from -2 to 2 in 5 increments

# Problem A.3
t = np.linspace(-2, 2, 8)  # defining 1000 points between -2 and 2
f_t = np.exp(-t) * np.cos(2 * np.pi * t)  # defining function
plot(f_t, t,figsize=(8.0, 4.0), title='f(t) = e^(-t) * cos(2πt) plot', plotLabel='e^(-t) * cos(2πt)', xLabel='t', yLabel='f(t)')
t = np.linspace(-2, 2, 5)  # 5 points from -2 to 2
f_t = np.exp(t)
plt.grid(True) #grid was not showing up until this 
plot(f_t, t, False, figsize=(8.0, 4.0), title='f(t) = e^(-t) plot', plotLabel='e^(-t)', xLabel='t', yLabel='f(t)')
plt.grid(True) #grid was not showing up until this 

# Problem B.1
t = np.linspace(-1,2, 1000) #defining 1000 points between -1 and 2 
p_t = np.heaviside(t, 1) - np.heaviside(t - 1, 1) #unit step function
plot(p_t, t, figsize=(8.0, 4.0), title='p(t) = u(t)−u(t −1) over (−1 ≤ t ≤ 2).', plotLabel='u(t)−u(t −1)', xLabel='t', yLabel='p(t)')

# Problem B.2
    # function for r(t) as its used more than once
def r(t):
    return t * p_t

r_t = r(t) #defining r_t
n_t = r(t) + r(-t +2) #defining n_t

    # plotting both on the same graph
plot(r_t,t,figsize=(8.0, 4.0), title='r(t) = tp(t)', plotLabel='r(t) = tp(t)', xLabel='t',yLabel='r(t)')
plot(n_t,t, newGraph=False, figsize=(8,4), title='r(t) = tp(t) & n(t) =r(t) + r(−t + 2).', plotLabel='n(t) = r(t) + r(−t + 2).', xLabel='t',yLabel='n(t)')
plt.grid(True) #plot grid wasnt showing up found this solution

# Problem B.3



plt.show()  # show all the plots (last line)