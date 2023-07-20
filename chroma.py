audio='AudioClip.wav' #location

import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load an audio file
y, sr = librosa.load(audio)

# Compute chroma features
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Display the chroma features
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()