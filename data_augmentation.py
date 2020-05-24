import numpy as np
import librosa

def add_random_noise(audio_signal, mu=0, stdev= 0.05):
    random_noise = np.random.normal(mu, stdev, len(audio_signal))
    return audio_signal + random_noise

def augment_audio_with_random_noise(audio_signal, min_stdev, max_stdev, n):
    stdevs = np.linspace(min_stdev, max_stdev, n)
    augmented_noise_audio_tracks = [add_random_noise(audio_signal, 0, stdev) for stdev in stdevs]
    return augmented_noise_audio_tracks

def change_pitch(audio_signal, sampling_rate, pitch_step):
    audio_pitch_shift = librosa.effects.pitch_shift(audio_signal, sampling_rate, pitch_step)
    return audio_pitch_shift

def augment_audio_with_pitch_shift(audio_signal, sampling_rate, min_pitch_shift, max_pitch_shift, n):
    pitch_steps = np.linspace(min_pitch_shift, max_pitch_shift, n)
    augmented_pitch_shift_audio_tracks = [change_pitch(audio_signal, sampling_rate, step) for step in pitch_steps]
    return augmented_pitch_shift_audio_tracks