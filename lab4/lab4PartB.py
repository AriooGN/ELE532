import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from MagSpect import MagSpect
from numpy import fft
import sounddevice as sd
import time

# Path to your MATLAB file
file_path = "lab4\Lab4_Data.mat"

# Load the MATLAB file
mat_data  = scipy.io.loadmat(file_path)

# Extracting the data
xspeech = np.squeeze(mat_data['xspeech'])
hLPF2000 = np.squeeze(mat_data['hLPF2000'])
hLPF2500 = np.squeeze(mat_data['hLPF2500'])
hChannel = np.squeeze(mat_data['hChannel'])
Fs = mat_data['Fs'].item() 


# Analyzing HChannel
# Impulse Response
plt.figure()
plt.plot(hChannel)
plt.title('Impulse Response of hChannel')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.savefig('lab4\Initial Analysis\hChannel_impulse_response.png')  # Save the figure as a PNG
plt.show()
# Frequency Response
MagSpect(hChannel)
plt.title('Frequency Response of hChannel')
plt.savefig('lab4\Initial Analysis\hChannel_frequency_response.png')  # Save the figure as a PNG
plt.show()

# Analyzing hLPF2000
# Impulse Response
plt.figure()
plt.plot(hLPF2000)
plt.title('Impulse Response of hLPF2000')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.savefig('lab4\Initial Analysis\hLPF2000_impulse_response.png')  # Save the figure as a PNG
plt.show()

# Frequency Response
MagSpect(hLPF2000)
plt.title('Frequency Response of hChannel')
plt.savefig('lab4\Initial Analysis\hLPF2000_frequency_response.png')  # Save the figure as a PNG
plt.show()

# Analyzing hLPF2500
# Impulse Response
plt.figure()
plt.plot(hLPF2500)
plt.title('Impulse Response of hLPF2500')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.savefig('lab4\Initial Analysis\hLPF2500_impulse_response.png')  # Save the figure as a PNG
plt.show()

# Frequency Response
MagSpect(hLPF2500)
plt.savefig('lab4\Initial Analysis\hLPF2500_frequency_response.png')  # Save the figure as a PNG
plt.show()

# Analyzing xSpeech
# Impulse Response
plt.figure()
plt.plot(xspeech)
plt.title('Impulse Response of xSpeech')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.savefig('lab4\Initial Analysis\Speechx_Impulse_response.png')
MagSpect(xspeech)
plt.savefig('lab4\Initial Analysis\Speechx_frequency_response.png')



# Encoding Process


# Filtering 
xspeech_filtered = np.convolve(xspeech, hLPF2000, mode='same')
MagSpect(xspeech_filtered)
plt.title("Magnitude Spectrum of Filtered Signal")
plt.savefig("lab4\Coding\Coder_Filtered_Signal.png")

# Shifting
shift_frequency = 5000  # Frequency shift in Hz
t = np.arange(len(xspeech_filtered)) / Fs
print(t)
xspeech_shifted = xspeech_filtered * np.exp(2j * np.pi * shift_frequency * t)
MagSpect(xspeech_shifted)
plt.title("Magnitude Spectrum of Shifted Signal")
plt.savefig("lab4\Coding\Coder_Shifted_Signal.png")

# Transmitting
transmitted_signal = np.convolve(xspeech_shifted, hChannel, mode='same')
MagSpect(transmitted_signal)
plt.title("Magnitude Spectrum of Transmited Signal")
plt.savefig("lab4\Coding\Transmited_Signal.png")


# Deconding Proccess

# Frequency Demodulation (removing removing shift)
t = np.arange(len(transmitted_signal)) / Fs # Time vector based on 'received sinal'
received_signal_demodulated = transmitted_signal * np.exp(-2j * np.pi * shift_frequency * t) 
MagSpect(received_signal_demodulated)
plt.title("Magnitude Spectrum of Demodulated Signal")
plt.savefig("lab4\Decoding\output_signal_demodulated.png")

# Post-filtering (applying a filter and removing excess noise picked up from channel)
recovered_signal_filtered = np.convolve(received_signal_demodulated, hLPF2500, mode='same')
MagSpect(recovered_signal_filtered)
plt.title("Magnitude Spectrum of Post-Filtered Signal")
plt.savefig("lab4\Decoding\output_signal_filtered.png")

# Normalization (Scaling makes it so it peaks @1

recovered_signal_normalized = recovered_signal_filtered / np.max(np.abs(recovered_signal_filtered))
MagSpect(recovered_signal_normalized)
plt.title("Magnitude Spectrum of Post-Normalized Signal")
plt.savefig("lab4\Decoding\output_signal_normalized.png")

print('Playing 1st Audio')
sd.play(np.real(xspeech),Fs)
sd.wait()

print('Playing 2nd Audio')
sd.play(np.real(recovered_signal_normalized),Fs)
sd.wait()