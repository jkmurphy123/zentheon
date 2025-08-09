import numpy as np
from audio_io import play

t = 1.0       # seconds
sr = 16000    # sample rate
f = 880.0     # tone frequency (A5)

x = np.sin(2 * np.pi * np.arange(int(t * sr)) * f / sr).astype("float32")
play(x, sr)
