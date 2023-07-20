audio='AudioClip.wav' #location

import librosa
import librosa.display
import matplotlib.pyplot as plt


# Load an audio file
y, sr = librosa.load(audio)

# Compute spectral contrast
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

# Display the spectral contrast
plt.figure(figsize=(10, 4))
librosa.display.specshow(contrast, x_axis='time')
plt.colorbar(label='Spectral Contrast')
plt.title('Spectral Contrast')
plt.tight_layout()
plt.show()