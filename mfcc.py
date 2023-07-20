audio='AudioClip.wav' #location

import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load an audio file
y, sr = librosa.load(audio)

# Compute MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)

# Display the MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()