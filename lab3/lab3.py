import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import cmath
from sympy import symbols, cos, pi, exp, integrate


def Dn(x, n):
    t = symbols('t')
    t0_1 = 20
    w0_1 = 2 * np.pi / t0_1

    if x < 0 or x > 2:
        return None

    n_array = np.array(n)
    Dn_array = np.zeros_like(n_array, dtype=complex)
    
    if x == 0:
        Dn_array[n_array == 1] = 1/4
        Dn_array[n_array == -1] = 1/4
        Dn_array[n_array == 3] = 1/2
        Dn_array[n_array == -3] = 1/2

        return Dn_array

    elif x == 1 or x == 2:
        # Handle division by zero
        n_non_zero = np.where(n_array != 0, n_array, np.nan)  # Replace 0 with nan
        Dn_array = 1 / (n_non_zero * np.pi)
        
        if x == 1:
            Dn_array *= np.sin((n_non_zero * np.pi) / 2)
        elif x == 2:
            Dn_array *= np.sin((n_non_zero * np.pi) / 4)
        
        # Handle the case when n = 0 (replace with the correct value if needed)
        Dn_array[np.isnan(Dn_array)] = 0  # Replace nan with 0
        return Dn_array

def plot_spectra(D_n, n_range, title):
    plt.figure(figsize=(12, 5))
    
    # Plot magnitude spectrum
    plt.subplot(1, 2, 1)
    plt.stem(n_range, np.abs(D_n), 'k', markerfmt='ok')
    plt.xlabel('n')
    plt.ylabel('|D_n|')
    plt.title(f'Magnitude Spectrum of {title}')
    
    # Plot phase spectrum
    plt.subplot(1, 2, 2)
    plt.stem(n_range, np.angle(D_n), 'k', markerfmt='ok')
    plt.xlabel('n')
    plt.ylabel('∠ D_n [rad]')
    plt.title(f'Phase Spectrum of {title}')
    
    plt.tight_layout()


# Part A.4
ranges = [list(range(-5, 6)), list(range(-20, 21)), list(range(-50, 51)), list(range(-500, 501))]
int_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    
for i in range(0, 4):
    currentRange = ranges[i]
    for j in range(0, 3):
        Dn_x = Dn(j, currentRange)
        if Dn_x is not None:
            n = np.array(currentRange)
            title = f'x{j+1} ({n[0]} ≤ n ≤ {n[-1]})'
            plot_spectra(Dn_x, n, title)
            plt.show()

# Part A.5
def reconstruct_signal(Dn, n_range, t):
    w0 = 2 * np.pi / 20  # Assuming T0 is 20 for x1(t), adjust as needed for other signals
    x_reconstructed = np.zeros_like(t, dtype=complex)
    
    for n, D in zip(n_range, Dn):   
        x_reconstructed += D * np.exp(1j * n * w0 * t)
    
    return x_reconstructed.real


# Part A.6

def reconstruct_signal(Dn, n_range, t):
    w0 = 2 * np.pi / 20  # Assuming T0 is 20 for x1(t), adjust as needed for other signals
    x_reconstructed = np.zeros_like(t, dtype=complex)
    
    for n, D in zip(n_range, Dn):
        x_reconstructed += D * np.exp(1j * n * w0 * t)
    
    return x_reconstructed.real

# Define the time vector for reconstruction
t = np.arange(-300, 301)  # t from -300 to 300

# Reconstruct and plot signals for different ranges of n
for i, n_range in enumerate(ranges):
    plt.figure(figsize=(14, 8))
    
    for j in range(3):
        Dn_x = Dn(j, n_range)
        if Dn_x is not None:
            x_reconstructed = reconstruct_signal(Dn_x, n_range, t)
            print(x_reconstructed)
            plt.subplot(3, 1, j+1)
            plt.plot(t, x_reconstructed, label=f'x{j+1}(t) Reconstructed')
            plt.xlabel('t (sec)')
            plt.ylabel(f'x{j+1}(t)')
            plt.title(f'Reconstructed x{j+1}(t) from Fourier Coefficients (n={n_range[0]} to {n_range[-1]})')
            plt.grid()
            plt.legend()
            plt.show()
    
    plt.tight_layout()



