audio='AudioClip.wav' #location

import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load an audio file
y, sr = librosa.load(audio)

# Compute harmonic component of the audio signal
y_harmonic = librosa.effects.harmonic(y)

# Compute tonnetz
tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

# Display the tonnetz
plt.figure(figsize=(10, 4))
librosa.display.specshow(tonnetz, x_axis='time', y_axis='tonnetz')
plt.colorbar()
plt.title('Tonal Centroids (Tonnetz)')
plt.tight_layout()
plt.show()