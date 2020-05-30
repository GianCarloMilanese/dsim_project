import numpy as np
import librosa
import os

MIN_STDEV = 0.001
MAX_STDEV = 0.05
MIN_PITCH_STEP = -6
MAX_PITCH_STEP = 5


def add_random_noise(audio_signal, mu=0, stdev=0.05):
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


def enrich_dataset(audio_dir, mode, n_noise, n_pitch, rate=8000):
    enriched_audio_tracks = {}
    for audio_fn in os.listdir(audio_dir):
        # Skip temporary files
        if audio_fn.endswith(".wav"):
            # _, original_signal = wav.read(os.path.join(audio_dir,audio_fn))
            original_signal, _ = librosa.core.load(os.path.join(audio_dir, audio_fn), rate)
            # Create an empty dict for storing the various tracks associated with the current file
            enriched_audio_tracks[audio_fn] = {}
            # Add the current audio
            enriched_audio_tracks[audio_fn]['original'] = [original_signal]
            # Apply various random noises to the original track
            noise_tracks = augment_audio_with_random_noise(original_signal, MIN_STDEV, MAX_STDEV,
                                                           n_noise)
            # Add these tracks to the result dictionary
            enriched_audio_tracks[audio_fn]['noise'] = noise_tracks
            # Apply pitch shift only to the original audio
            current_pitch_tracks = augment_audio_with_pitch_shift(audio_signal=original_signal,
                                                                  sampling_rate=rate,
                                                                  min_pitch_shift=MIN_PITCH_STEP,
                                                                  max_pitch_shift=MAX_PITCH_STEP,
                                                                  n=n_pitch)
            # Store them
            enriched_audio_tracks[audio_fn]['pitch'] = current_pitch_tracks
            if mode == "all_combinations":
                # Create a list for storing the tracks obtained through pitch shift
                pitch_noise_tracks = []
                # Iterate on the list with noise tracks
                for track in noise_tracks:
                    current_pitch_tracks = augment_audio_with_pitch_shift(track,
                                                                          rate,
                                                                          MIN_PITCH_STEP,
                                                                          MAX_PITCH_STEP,
                                                                          n_pitch)
                    pitch_noise_tracks = pitch_noise_tracks + current_pitch_tracks
                # Add the tracks enriched with pitch shift to the current list
                enriched_audio_tracks[audio_fn]['pitch_noise'] = pitch_noise_tracks

    return enriched_audio_tracks
