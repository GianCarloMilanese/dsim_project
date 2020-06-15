import numpy as np
import librosa
import os

MIN_PITCH_STEP = -6
MAX_PITCH_STEP = 5
RATE = 8000


def add_random_noise(audio_signal, mu=0, stdev=0.05):
    """
    Add random noise to the given track
    :param audio_signal:
    :param mu:
    :param stdev:
    :return:
    """
    random_noise = np.random.normal(mu, stdev, len(audio_signal))
    return audio_signal + random_noise


def augment_audio_with_random_noise(audio_signal, min_stdev, max_stdev, n):
    """
    Create n recordings with various levels of random noise
    :param audio_signal: Audio track to augment
    :param min_stdev: min standard deviation for adding noise
    :param max_stdev: max standard deviation for adding noise
    :param n: number of desired augmented recordings
    :return:
    """
    stdevs = np.linspace(min_stdev, max_stdev, n)
    augmented_noise_audio_tracks = [add_random_noise(audio_signal, 0, stdev) for stdev in stdevs]
    return augmented_noise_audio_tracks


def change_pitch(audio_signal, sampling_rate, pitch_step):
    """
    Modify the pitch of the given recording
    :param audio_signal:
    :param sampling_rate:
    :param pitch_step:
    :return:
    """
    audio_pitch_shift = librosa.effects.pitch_shift(audio_signal, sampling_rate, pitch_step)
    return audio_pitch_shift


def augment_audio_with_pitch_shift(audio_signal, sampling_rate, min_pitch_shift, max_pitch_shift, n):
    """
    Create n recordings with various levels of pitch.
    :param audio_signal: Audio track to augment
    :param sampling_rate: sampling rate of input audio
    :param min_pitch_shift: minimum pitch shift value
    :param max_pitch_shift: maximum pitch shift value
    :param n: number of desidered augmented recordings
    :return:
    """
    pitch_steps = np.linspace(min_pitch_shift, max_pitch_shift, n)
    augmented_pitch_shift_audio_tracks = [change_pitch(audio_signal, sampling_rate, step) for step in pitch_steps]
    return augmented_pitch_shift_audio_tracks


def enrich_dataset(audio_dir: str, mode: str, n_noise: int, n_pitch: int, recordings_made_by_us: bool, max_length=999999):
    """
    Augment all recordings in the target directory
    :param audio_dir: path where audio tracks are stored
    :param mode: whether to apply data augmentation strategies sequentially or in a combinatorial way
    :param n_noise: number of "random noise" tracks to produce from one recording
    :param n_pitch: number of "modified pitch" tracks to produce from one recording
    :param max_length: maximum length of a recording should have in order to be considered
    :param recordings_made_by_us: whether the recording is made by us or not
    :return:
    """
    print("enrich_dataset>>>")
    if recordings_made_by_us:
        MAX_STDEV = 0.05
        MIN_STDEV = 0.002
    else:
        MAX_STDEV = 0.025
        MIN_STDEV = 0.001
    enriched_audio_tracks = {}
    for audio_fn in os.listdir(audio_dir):
        # Skip temporary files
        if audio_fn.endswith(".wav"):
            # _, original_signal = wav.read(os.path.join(audio_dir,audio_fn))
            original_signal, _ = librosa.load(os.path.join(audio_dir, audio_fn), sr=RATE)
            if len(original_signal) > max_length:
                print("Max length: {}, shape:{}".format(max_length, original_signal.shape))
                next
            else:
                # Create an empty dict for storing the various tracks associated with the current file
                enriched_audio_tracks[audio_fn] = {}
                # Add the current audio
                enriched_audio_tracks[audio_fn]['original'] = [original_signal]
                # Apply various random noises to the original track
                noise_tracks = augment_audio_with_random_noise(original_signal, MIN_STDEV, MAX_STDEV,
                                                               n_noise)
                # Add these tracks to the result dictionary
                enriched_audio_tracks[audio_fn]['noise'] = noise_tracks
                if n_pitch > 0:
                    # Apply pitch shift only to the original audio
                    current_pitch_tracks = augment_audio_with_pitch_shift(audio_signal=original_signal,
                                                                          sampling_rate=RATE,
                                                                          min_pitch_shift=MIN_PITCH_STEP,
                                                                          max_pitch_shift=MAX_PITCH_STEP,
                                                                          n=n_pitch)
                    # Store them
                    enriched_audio_tracks[audio_fn]['pitch'] = current_pitch_tracks
                if mode == "all_combinations":
                    # Create a list for storing the tracks obtained through pitch shift
                    pitch_noise_tracks = []
                    if n_pitch > 0:
                        # Iterate on the list with noise tracks
                        for track in noise_tracks:
                            current_pitch_tracks = augment_audio_with_pitch_shift(track,
                                                                                  RATE,
                                                                                  MIN_PITCH_STEP,
                                                                                  MAX_PITCH_STEP,
                                                                                  n_pitch)
                            pitch_noise_tracks = pitch_noise_tracks + current_pitch_tracks
                        # Add the tracks enriched with pitch shift to the current list
                        enriched_audio_tracks[audio_fn]['pitch_noise'] = pitch_noise_tracks
    print("enrich_dataset <<<")
    return enriched_audio_tracks
