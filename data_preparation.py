from tensorflow import keras
from sklearn.model_selection import train_test_split, GridSearchCV
import librosa
import numpy as np
from tqdm.notebook import tqdm
import os


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
                    audio, sample_rate = librosa.load(path + "/" + f)
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


def pad_zeros(recordings):
    max_y = max(map(np.shape, recordings))[0]
    res = []
    for rec in recordings:
        diff_in_y = max_y - rec.shape[0]
        if diff_in_y > 0:
            half_diff = int(diff_in_y/2)
            remaining_diff = diff_in_y-half_diff
            v = np.pad(rec, (half_diff, remaining_diff), 'constant', constant_values=0)
            res.append(v)
        else:
            res.append(rec)
    return res


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
    X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def split_train_test_nn(X, y, test_size=0.2, number_mode=True):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    if number_mode:
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
    return X_train, X_test, y_train, y_test, input_shape


def transform_categorical_y(labels):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    label_0 = enc.inverse_transform(np.array([0, 0, 0, 0, 0, 0, 0]).reshape(1, -1))[0][0]
    label_1 = enc.inverse_transform(np.array([0, 1, 0, 0, 0, 0, 0]).reshape(1, -1))[0][0]
    label_2 = enc.inverse_transform(np.array([0, 0, 1, 0, 0, 0, 0]).reshape(1, -1))[0][0]
    label_3 = enc.inverse_transform(np.array([0, 0, 0, 1, 0, 0, 0]).reshape(1, -1))[0][0]
    label_4 = enc.inverse_transform(np.array([0, 0, 0, 0, 1, 0, 0]).reshape(1, -1))[0][0]
    label_5 = enc.inverse_transform(np.array([0, 0, 0, 0, 0, 1, 0]).reshape(1, -1))[0][0]
    label_6 = enc.inverse_transform(np.array([0, 0, 0, 0, 0, 0, 1]).reshape(1, -1))[0][0]
    target_names = [label_0, label_1, label_2, label_3, label_4, label_5, label_6]
    return y, target_names
