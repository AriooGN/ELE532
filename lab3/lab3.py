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
        # Define x_1(t) as a lambda function
        x_1 = lambda t: np.cos(3 * np.pi / 10 * t) + 0.5 * np.cos(np.pi / 10 * t)

        # Perform the integral for each n
        for i, n_val in enumerate(n):
            integral_result, _ = quad(lambda t: x_1(t) * cmath.exp(-1j * w0_1 * t * n_val).real, 0, t0_1)
            Dn_array[i] = integral_result / t0_1
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
    plt.show()

# Part A.4
ranges = [list(range(-5, 6)), list(range(-20, 21)), list(range(-50, 51)), list(range(-500, 501))]
int_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

for i in range(0, 4):
    currentRange = ranges[i]
    for j in range(0, 3):
        Dn_x = Dn(j, currentRange)
        if Dn_x is not None:
            n = np.array(currentRange)
            #title = f'x' + str(j+1) + ' (' + str(n[0]) +' ≤ n ≤ ' + str(n[-1]) + ')'
            title = f'x{j+1} ({n[0]} ≤ n ≤ {n[-1]})'
            plot_spectra(Dn_x, n, title)

