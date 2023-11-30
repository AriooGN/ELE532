import numpy as np
import matplotlib.pyplot as plt
"""
Arian Fooladray
501112069
This Function is my adapdations of the provided filter of the MagSpect.m. The description is borrowed from author.
The original authur is : AUTHOR  : M. Zeytinoglu 
                                   Department of Electrical & Computer Engineering
                                   Ryerson University
                                   Toronto, Ontario, CANADA
"""


def MagSpect(x, Fs=32000):
    """
    MagSpect ... Utility function to simplify plotting the magnitude spectrum.

    MagSpect(x) plots the double-sided magnitude spectrum of x using 
    a 1024-point FFT; the frequency axis labels are generated based 
    on the sampling frequency Fs (default is 32 kHz). Spectral magnitude
    values are plotted in dB.

    Version: 1.0
    Date: November 2023.
    """
    Nfft = 1024  # Default FFT size

    # Set up the frequency vector
    ff = np.fft.fftfreq(Nfft, 1 / Fs)
    ff = np.fft.fftshift(ff)  # Shift zero frequency to center

    # Compute the spectrum of x(t) using Nfft-point FFT
    Xspect = np.fft.fft(x, Nfft)
    Xspect = np.fft.fftshift(Xspect)  # Shift zero frequency to center

    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(ff, 20 * np.log10(np.abs(Xspect)))
    plt.xlim([-Fs / 2, Fs / 2])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Magnitude Spectrum')
    plt.grid(True)

