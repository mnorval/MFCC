audio='AudioClip.wav' #location

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an audio file
y, sr = librosa.load(audio)

# Compute a mel-scaled spectrogram
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to log scale (dB)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

# Display the mel-scaled spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spect, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-scaled Spectrogram')
plt.tight_layout()
plt.show()