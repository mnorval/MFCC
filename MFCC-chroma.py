import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import librosa







TRAIN_PATH = 'D:\\PhD\\Afrikaans\\Anger\\'
#D:\Google Drive (mnorval@gmail.com)\Personal\Study\Phd\Audio Speech Samples\Afrikaans\Anger
sample_rate, signal = wavfile.read(TRAIN_PATH + "[Angry Farmer phones Eskom] - 21.wav")

signal = signal[0:int(10* sample_rate)]
Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
eps = np.finfo(float).eps
import librosa, librosa.display
C = librosa.feature.chroma_stft(y=signal, sr=sample_rate, tuning=0, norm=None, hop_length=Time, n_fft=16)
plt.figure(figsize=(8, 2))
librosa.display.specshow(10 * np.log10(eps + C), x_axis='time', 
                         y_axis='chroma', sr=sample_rate, hop_length=16)
plt.colorbar();
"""
#FRAMING
#FRAMING
frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

#plt.subplot(4, 1, 1)
#plt.title('Original Time Domain Signal')
#plt.plot(Time, signal)
#plt.subplot(3, 1, 1)
#plt.title('Framing - Original Frame 1')
#plt.plot(frames[50]) 

#plt.subplot(3, 1, 3)
#plt.title('Framing - Combined')
#plt.plot(frames[50])

#plt.subplot(4, 1, 3)
#plt.title('Framing - Sample Frame 2')
#plt.plot(frames[60],'tab:green') 

#plt.subplot(4, 1, 4)
#plt.title('Framing - Sample Frame 3')
#plt.plot(frames[70],'tab:pink') 

#plt.tight_layout()
#plt.show()

frames *= np.hamming(frame_length)
# frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **

#plt.subplot(3, 1, 2)
#plt.title('Framing - Windowed Frame 1')
#plt.plot(frames[50],'tab:orange') 

#plt.subplot(3, 1, 3)
#plt.title('Framing - Combined')
#plt.plot(frames[50])

#plt.subplot(4, 1, 3)
#plt.title('Framing - Sample Frame 2')
#plt.plot(frames[60],'tab:green') 

#plt.subplot(4, 1, 4)
#plt.title('Framing - Sample Frame 3')
#plt.plot(frames[70],'tab:pink') 


#plt.show()

NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum


N = 512
T = 1.0 / len(frames[50])

#x = np.linspace(0.0, N*T, N)
#y = np.sin(60.0 * 2.0*np.pi*x) + 0.5*np.sin(90.0 * 2.0*np.pi*x)
x = np.linspace(0.0, N*T, N)
y = frames[50]


y_f = np.fft.fft(y)
x_f = np.linspace(0.0, 1.0/(2.0*T), N//2)


plt.subplot(2, 1, 1)
plt.title('Frame 50')
plt.plot(frames[50])

plt.subplot(2, 1, 2)
plt.title('Frame 50 - FFT')
plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
plt.tight_layout()
"""
plt.show()
