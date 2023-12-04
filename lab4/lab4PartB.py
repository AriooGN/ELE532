import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from MagSpect import MagSpect  

# Path to MATLAB file
file_path = "lab4/Lab4_Data.mat"

# Load the MATLAB file
mat_data = scipy.io.loadmat(file_path)

# Extracting the data
xspeech = np.squeeze(mat_data['xspeech'])
hLPF2000 = np.squeeze(mat_data['hLPF2000'])
hLPF2500 = np.squeeze(mat_data['hLPF2500'])
hChannel = np.squeeze(mat_data['hChannel'])
Fs = mat_data['Fs'].item()

# Function to plot and save a figure
def plot_and_save(data, title, filename):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.savefig(filename)
    plt.show()

# Analyzing hChannel
# Impulse Response
plot_and_save(hChannel, 'Impulse Response of hChannel', 'lab4/Initial Analysis/hChannel_impulse_response.png')

# Frequency Response
MagSpect(hChannel)
plt.title('Frequency Response of hChannel')
plt.savefig('lab4/Initial Analysis/hChannel_frequency_response.png')
plt.show()

# Analyzing hLPF2000
# Impulse Response
plot_and_save(hLPF2000, 'Impulse Response of hLPF2000', 'lab4/Initial Analysis/hLPF2000_impulse_response.png')

# Frequency Response
MagSpect(hLPF2000)
plt.title('Frequency Response of hLPF2000')
plt.savefig('lab4/Initial Analysis/hLPF2000_frequency_response.png')
plt.show()

# Analyzing hLPF2500
# Impulse Response
plot_and_save(hLPF2500, 'Impulse Response of hLPF2500', 'lab4/Initial Analysis/hLPF2500_impulse_response.png')

# Frequency Response
MagSpect(hLPF2500)
plt.title('Frequency Response of hLPF2500')
plt.savefig('lab4/Initial Analysis/hLPF2500_frequency_response.png')
plt.show()

# Analyzing xSpeech
# Impulse Response
plot_and_save(xspeech, 'Impulse Response of xSpeech', 'lab4/Initial Analysis/Speechx_Impulse_response.png')

# Frequency Response
MagSpect(xspeech)
plt.title('Frequency Response of xSpeech')
plt.savefig('lab4/Initial Analysis/Speechx_frequency_response.png')
plt.show()

# Encoding Process

# Filtering
xspeech_filtered = np.convolve(xspeech, hLPF2000, mode='same')
MagSpect(xspeech_filtered)
plt.title("Magnitude Spectrum of Filtered Signal")
plt.savefig("lab4/Coding/Coder_Filtered_Signal.png")
plt.show()

# Shifting
shift_frequency = 5000  # Frequency shift in Hz
t = np.arange(len(xspeech_filtered)) / Fs
xspeech_shifted = xspeech_filtered * np.exp(2j * np.pi * shift_frequency * t)
MagSpect(xspeech_shifted)
plt.title("Magnitude Spectrum of Shifted Signal")
plt.savefig("lab4/Coding/Coder_Shifted_Signal.png")
plt.show()

# Transmitting
transmitted_signal = np.convolve(xspeech_shifted, hChannel, mode='same')
MagSpect(transmitted_signal)
plt.title("Magnitude Spectrum of Transmitted Signal")
plt.savefig("lab4/Coding/Transmitted_Signal.png")
plt.show()

# Decoding Process

# Frequency Demodulation (removing shift)
t = np.arange(len(transmitted_signal)) / Fs
received_signal_demodulated = transmitted_signal * np.exp(-2j * np.pi * shift_frequency * t)
MagSpect(received_signal_demodulated)
plt.title("Magnitude Spectrum of Demodulated Signal")
plt.savefig("lab4/Decoding/Output_Signal_Demodulated.png")
plt.show()

# Post-filtering (applying a filter and removing excess noise from channel)
recovered_signal_filtered = np.convolve(received_signal_demodulated, hLPF2500, mode='same')
MagSpect(recovered_signal_filtered)
plt.title("Magnitude Spectrum of Post-Filtered Signal")
plt.savefig("lab4/Decoding/Output_Signal_Filtered.png")
plt.show()

# Normalization (scaling to peak at 1)
recovered_signal_normalized = recovered_signal_filtered / np.max(np.abs(recovered_signal_filtered))
MagSpect(recovered_signal_normalized)
plt.title("Magnitude Spectrum of Post-Normalized Signal")
plt.savefig("lab4/Decoding/Output_Signal_Normalized.png")
plt.show()

# Audio Playback
print('Playing 1st Audio')
sd.play(np.real(xspeech), Fs)
sd.wait()

print('Playing 2nd Audio')
sd.play(np.real(recovered_signal_normalized), Fs)
sd.wait()
