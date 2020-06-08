from typing import List
from tensorflow import keras
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from tqdm.notebook import tqdm
import os
import data_augmentation
import random

RATE = 8000


def load_recordings(paths=["recordings"], label_type="number"):
    res = []
    for path in paths:
        print(f"Loading from {path}")
        for f in tqdm(sorted(os.listdir(path))):
            if f.endswith('.wav'):
                if "pitch" in f and label_type == "speaker":
                    # do not consider pitch alteration
                    next
                else:
                    # Load file and extract features
                    audio, sample_rate = librosa.load(path + "/" + f, sr=RATE)
                    res.append(audio)
    return np.array(res)


def load_labels(paths=["recordings"], label_type="number"):
    labels = []
    for path in paths:
        for f in sorted(os.listdir(path)):
            if f.endswith('.wav'):
                if "pitch" in f and label_type == "speaker":
                    next
                else:
                    if label_type.startswith("n"):
                        label = f.split('_')[0]
                    else:
                        label = f.split('_')[1]
                    labels.append(label)
    return labels


def pad_zeros(recordings, compute_max_y=True, max_y=0):
    print("pad_zeros >>>")
    if compute_max_y:
        max_y = max(map(np.shape, recordings))[0]
    res = []
    for rec in recordings:
        res.append(padding(max_y, rec))
    print("pad_zeros <<<")
    return np.array(res)


def padding(max_y, rec):
    diff_in_y = max_y - rec.shape[0]
    if diff_in_y > 0:
        half_diff = int(diff_in_y / 2)
        remaining_diff = diff_in_y - half_diff
        v = np.pad(rec, (half_diff, remaining_diff), 'constant', constant_values=0)
        return v
    else:
        return rec


def compute_spectrogram(audio, rate=8000, n_fft=1024, hop_length=160, n_mels=128, normalize=False):
    spectrogram = librosa.feature.melspectrogram(y=np.array(audio),
                                                 sr=rate,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels)
    if normalize:
        spectrogram = np.log10(1000 * spectrogram + 1)
    return spectrogram


def split_train_test_baseline_spectrograms(X, y):
    nsamples, nx, ny = X.shape
    X_2d = X.reshape((nsamples, nx * ny))
    # Get the training set : 60% of original data
    X_train, X_test_val, y_train, y_test_val = train_test_split(X_2d, y, test_size=0.4, random_state=1)
    # Get validation and test set, both 20% of original data
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_train_test_nn(X, y, number_mode=True):
    # Get the training set: 60% of original data
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.4, random_state=1)
    # Get val and test set, both 20% of original data
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=1)
    # Change shape of X for model training purpose
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    if number_mode:
        y_train = keras.utils.to_categorical(y_train, 10)
        y_val = keras.utils.to_categorical(y_val, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
    return X_train, X_val, X_test, y_train, y_val, y_test, input_shape


def transform_categorical_y(labels):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    label_0 = enc.inverse_transform(np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1))[0][0]
    label_1 = enc.inverse_transform(np.array([0, 1, 0, 0, 0, 0, 0, 0]).reshape(1, -1))[0][0]
    label_2 = enc.inverse_transform(np.array([0, 0, 1, 0, 0, 0, 0, 0]).reshape(1, -1))[0][0]
    label_3 = enc.inverse_transform(np.array([0, 0, 0, 1, 0, 0, 0, 0]).reshape(1, -1))[0][0]
    label_4 = enc.inverse_transform(np.array([0, 0, 0, 0, 1, 0, 0, 0]).reshape(1, -1))[0][0]
    label_5 = enc.inverse_transform(np.array([0, 0, 0, 0, 0, 1, 0, 0]).reshape(1, -1))[0][0]
    label_6 = enc.inverse_transform(np.array([0, 0, 0, 0, 0, 0, 1, 0]).reshape(1, -1))[0][0]
    label_7 = enc.inverse_transform(np.array([0, 0, 0, 0, 0, 0, 1, 0]).reshape(1, -1))[0][0]
    target_names = [label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7]
    return y, target_names


def get_list_indexes(lst, a, split_index):
    return [i for i, x in enumerate(lst) if a in "_" + x.split('_')[split_index] + "_"]


def split_and_augment_dataset(audio_dir: str,
                              y_type: str,
                              n_category_audio_to_pick_test: int,
                              include_pitch: bool,
                              max_length: int):
    print("split_and_augment_dataset >>>")
    augmented_tracks = data_augmentation.enrich_dataset(audio_dir,
                                                        mode="normal",
                                                        n_noise=5,
                                                        n_pitch=5,
                                                        max_length=max_length)
    if y_type == "speakers_us":
        categories = ['_gian_', '_alinda_', '_khaled_', '_ale_']
        # Used later on for getting y label
        split_index = 1
    elif y_type == "speakers_default":
        categories = ['jackson', 'nicolas', 'theo', 'yweweler']
        split_index = 1
    else:
        categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        split_index = 0
    all_keys = [k for k in augmented_tracks.keys()]
    test_labels = []
    # Get the list of recordings name that will compose our test set
    for c in categories:
        indexes_of_c = get_list_indexes(all_keys, c, split_index)
        records_to_pick = random.sample(indexes_of_c, n_category_audio_to_pick_test)
        current_test_categories = [all_keys[i] for i in records_to_pick]
        test_labels = test_labels + current_test_categories
    # Get the recordings for the test set
    test_recordings = []
    for k in test_labels:
        # store original recording
        current_key = augmented_tracks[k]
        test_recordings.append(current_key['original'][0])
        # eliminate the original + augmented tracks from the dataset
        del augmented_tracks[k]
    train_recordings = []
    train_labels = []
    for k in augmented_tracks.keys():
        # Original track
        train_labels.append(k)
        train_recordings.append(augmented_tracks[k]['original'][0])
        # Noise track
        noise_recordings = augmented_tracks[k]['noise']
        noise_labels = len(noise_recordings) * [k]
        train_labels = train_labels + noise_labels
        train_recordings = train_recordings + noise_recordings
        # Pitch tracks
        if include_pitch:
            pitch_recordings = augmented_tracks[k]['pitch']
            pitch_labels = len(pitch_recordings) * [k]
            train_labels = train_labels + pitch_labels
            train_recordings = train_recordings + pitch_recordings
    # Get final label. The file format is number_speaker_n.wav
    train_labels = [label.split('_')[split_index] for label in train_labels]
    test_labels = [label.split('_')[split_index] for label in test_labels]
    train_recordings, val_recordings, train_labels, val_labels = train_test_split(train_recordings,
                                                                                  train_labels,
                                                                                  test_size=0.2,
                                                                                  random_state=1)
    print("split_and_augment_dataset <<<")
    return train_recordings, train_labels, val_recordings, val_labels, test_recordings, test_labels


def prepare_augmented_recordings(audio_dirs: List[str],
                                 y_type: List[str],
                                 n_category_test: int,
                                 include_pitch: bool,
                                 max_length: int):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    for i, dir_path in enumerate(audio_dirs):
        train_recordings, train_labels, val_recordings, val_labels, test_recordings, test_labels = split_and_augment_dataset(dir_path,
                                                                                                 y_type[i],
                                                                                                 n_category_test,
                                                                                                 include_pitch,
                                                                                                 max_length)
        X_train = X_train + train_recordings
        y_train = y_train + train_labels
        X_val = X_val + val_recordings
        y_val = y_val + val_labels
        X_test = X_test + test_recordings
        y_test = y_test + test_labels
    X_train = [np.array(x) for x in X_train]
    y_train = [np.array(x) for x in y_train]
    X_val = [np.array(x) for x in X_val]
    y_val = [np.array(x) for x in y_val]
    X_test = [np.array(x) for x in X_test]
    y_test = [np.array(x) for x in y_test]
    print("conversion_done!")
    X_train, X_val, X_test = compute_spectrograms(X_train, X_val, X_test)
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)


def compute_spectrograms(X_train, X_val, X_test):
    print("compute_spectrograms >>>")
    # In order to normalise the length of recordings we have to define the maximum length of the various recordings
    max_length_rec = max(map(np.shape, X_train + X_val + X_test))[0]
    print(max_length_rec)
    X_train = pad_zeros(X_train, compute_max_y=False, max_y=max_length_rec)
    X_val = pad_zeros(X_val, compute_max_y=False, max_y=max_length_rec)
    X_test = pad_zeros(X_test, compute_max_y=False, max_y=max_length_rec)
    print("Padding done")
    # Now let's compute the spectrograms
    X_train = [compute_spectrogram(x, normalize=True) for x in X_train]
    X_val = [compute_spectrogram(x, normalize=True) for x in X_val]
    X_test = [compute_spectrogram(x, normalize=True) for x in X_test]
    print("compute_spectrograms <<<")
    return X_train, X_val, X_test
