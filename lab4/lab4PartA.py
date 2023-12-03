import numpy as np
import matplotlib.pyplot as plt
import os

def plot_signal(time, signal, title, x_label, y_label, subplot_position=None, label=None, color=None):
    if subplot_position:
        plt.subplot(subplot_position)

    plt.plot(time, signal, label=label, color=color if color else 'blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if label:
        plt.legend()

def perform_convolution(signal, time):
    convolved_signal = np.convolve(signal, signal, 'full')
    conv_time = np.linspace(2*time[0], 2*time[-1], 2*len(time)-1)
    return convolved_signal, conv_time


# Create the directory for saving figures if it doesn't exist
save_path = 'lab4\\PartA'
os.makedirs(save_path, exist_ok=True)

# Problem A.1
problem = "A.1"
x = lambda t: np.where((t >= 0) & (t < 10), 1, 0)
t = np.arange(-10, 30, 0.001)
x_t = x(t)
z_t, t_conv = perform_convolution(x_t, t)

plt.figure(figsize=(14, 6))
plot_signal(t, x_t, f'{problem} - Original Signal x(t)', 'Time', 'Amplitude', subplot_position=211, label='x(t)')
plot_signal(t_conv, z_t, f'{problem} - Convolution of x(t) with itself', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

# Save the figures with problem numbers
plt.savefig(f'{save_path}\\{problem}_Original_Signal_x_t.png')
plt.savefig(f'{save_path}\\{problem}_Convolution_x_t_with_itself.png')

# Problem A.2
problem = "A.2"
N, PulseWidth = 100, 10
x_t = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))
t = np.linspace(-10, 30, N)
x_w = np.fft.fft(x_t)
z_w = np.abs(x_w)**2

plt.figure(figsize=(14, 6))
plot_signal(t, x_t, f'{problem} - Signal x(t)', 'Time', 'Amplitude', subplot_position=211, label='x(t)')
plot_signal(t, z_w, f'{problem} - Signal |X(ω)|', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

# Save the figures with problem numbers
plt.savefig(f'{save_path}\\{problem}_Signal_X_omega.png')
plt.savefig(f'{save_path}\\{problem}_Signal_X_omega_abs.png')

# Problem A.3
problem = "A.3"
N, PulseWidth = 100, 10
t = np.linspace(0, 1, N)
f = np.fft.fftfreq(N)
f = np.fft.fftshift(f)
z_w_abs = np.abs(z_w)
z_w_ang = np.angle(z_w)

plt.figure(figsize=(14, 6))
plot_signal(f, np.fft.fftshift(z_w), f'{problem} - Signal Z(ω)', 'Frequency', 'Amplitude', subplot_position=311, label='w(ω)')
plot_signal(f, np.fft.fftshift(z_w_abs), f'{problem} - Signal |Z(ω)|', 'Frequency', 'Amplitude', subplot_position=312, label='w(ω)')
plot_signal(f, np.fft.fftshift(z_w_ang), f'{problem} - Signal ∠Z(ω)', 'Frequency', 'Amplitude', subplot_position=313, label='w(ω)')
plt.tight_layout()

# Save the figures with problem numbers
plt.savefig(f'{save_path}\\{problem}_Signal_Z_omega.png')
plt.savefig(f'{save_path}\\{problem}_Signal_Z_omega_abs.png')
plt.savefig(f'{save_path}\\{problem}_Signal_Z_omega_angle.png')

# Problem A.4
problem = "A.4"
z_w_ifft = np.fft.ifftn(z_w)    
z_w_ifft = np.fft.ifftshift(z_w_ifft)
t = np.linspace(-10, 30, len(z_w_ifft))

plt.figure(figsize=(14, 6))
plot_signal(t_conv, z_t, f'{problem} - z(t) -> conv.', 'Time', 'Amplitude', subplot_position=211, label='z(t)')
plot_signal((t), z_w_ifft, f'{problem} - z(t) -> ifft', 'Time', 'Amplitude', subplot_position=212, label='z(t)')
plt.tight_layout()

# Save the figures with problem numbers
plt.savefig(f'{save_path}\\{problem}_z_t_conv.png')
plt.savefig(f'{save_path}\\{problem}_z_t_ifft.png')

# Problem A.5
problem = "A.5"
PulseWidths = [5, 10, 25]
N = 100
for i in range(0, 3):
    PulseWidth = PulseWidths[i]
    x_t = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))
    x_w = np.fft.fft(x_t)
    f = np.fft.fftfreq(N)
    f = np.fft.fftshift(f)
    x_w_abs = np.abs(x_w)
    x_w_ang = np.angle(x_w)
    
    plt.figure(figsize=(14, 6))
    plot_signal(f, np.fft.fftshift(x_w), f'{problem} - Signal X(ω), Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=311, label='w(ω)')
    plot_signal(f, np.fft.fftshift(x_w_abs), f'{problem} - Signal |X(ω)|, Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=312, label='w(ω)')
    plot_signal(f, np.fft.fftshift(x_w_ang), f'{problem} - Signal ∠X(ω), Pulse Width: {PulseWidths[i]}', 'Frequency', 'Amplitude', subplot_position=313, label='w(ω)')
    plt.tight_layout()

    # Save the figures with problem numbers
    plt.savefig(f'{save_path}\\{problem}_Signal_X_omega_{PulseWidths[i]}.png')

# Problem A.6
problem = "A.6"
N = 100
PulseWidth = 10
t = np.arange(0, N)
x = np.concatenate((np.ones(PulseWidth), np.zeros(N - PulseWidth)))

wp = x * np.exp(1j * (np.pi/3) * t)
wm = x * np.exp(-1j * (np.pi/3) * t)
wc = x * np.cos((np.pi/3) * t)


Xf = np.fft.fft(wp)
Yf = np.fft.fft(wm)
Zf = np.fft.fft(wc)


f = np.fft.fftfreq(N, 1/N)
f_shifted = np.fft.fftshift(f)



plt.figure(figsize=(14, 8))
plot_signal(f_shifted, np.fft.fftshift(Xf), f'{problem} - X(ω)', 'Frequency (ω)', 'Amplitude', subplot_position=311)
plot_signal(f_shifted, np.fft.fftshift(np.abs(Xf)), f'{problem} - |X(ω)|', 'Frequency (ω)', 'Magnitude', subplot_position=312)
plot_signal(f_shifted, np.fft.fftshift(np.angle(Xf)), f'{problem} - Angle X(ω)', 'Frequency (ω)', 'Phase (radians)', subplot_position=313)
plt.tight_layout()
plt.savefig(f'{save_path}/{problem}_Xf.png')


plt.figure(figsize=(14, 8))
plot_signal(f_shifted, np.fft.fftshift(Yf), f'{problem} - Y(ω)', 'Frequency (ω)', 'Amplitude', subplot_position=311)
plot_signal(f_shifted, np.fft.fftshift(np.abs(Yf)), f'{problem} - |Y(ω)|', 'Frequency (ω)', 'Magnitude', subplot_position=312)
plot_signal(f_shifted, np.fft.fftshift(np.angle(Yf)), f'{problem} - Angle Y(ω)', 'Frequency (ω)', 'Phase (radians)', subplot_position=313)
plt.tight_layout()
plt.savefig(f'{save_path}/{problem}_Yf.png')


plt.figure(figsize=(14, 8))
plot_signal(f_shifted, np.fft.fftshift(Zf), f'{problem} - Z(ω)', 'Frequency (ω)', 'Amplitude', subplot_position=311)
plot_signal(f_shifted, np.fft.fftshift(np.abs(Zf)), f'{problem} - |Z(ω)|', 'Frequency (ω)', 'Magnitude', subplot_position=312)
plot_signal(f_shifted, np.fft.fftshift(np.angle(Zf)), f'{problem} - Angle Z(ω)', 'Frequency (ω)', 'Phase (radians)', subplot_position=313)
plt.tight_layout()
plt.savefig(f'{save_path}/{problem}_Zf.png')

plt.show()
