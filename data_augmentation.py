import numpy as np

def add_random_noise(audio_signal, mu=0, stdev= 0.05):
    random_noise = np.random.normal(mu, stdev, len(audio_signal))
    return audio_signal + random_noise

def change_pitch(audio_signal, sampling_rate, pitch_step):
    audio_pitch_shift = librosa.effects.pitch_shift(audio_signal, sampling_rate, pitch_step)
    return audio_pitch_shift