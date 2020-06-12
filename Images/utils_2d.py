#!/usr/bin/env python3

# TODO:
# - Add docstrings

import os
import numpy as np
import shutil
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
from imutils.face_utils import rect_to_bb
import json


def take_bgr_picture():
    """
    Take a single picture

    :return: BGR image
    """
    cap = cv.VideoCapture(0)
    result, img = cap.read()
    img = cv.flip(img, 1)
    cap.release()
    return img


def take_rgb_picture():
    """
    Take a single picture

    :return: RGB image
    """
    return rgb_bgr_switch(take_bgr_picture())


def rgb_bgr_switch(img):
    """
    Switch between RGB and BGR

    :img: RGB/BGR image

    :return: BGR image if :img: is RGB, and viceversa
    """
    return img[:, :, ::-1]


def create_mask(a=92, b=112, n=224, r=112):
    """
    Create ellipsoidal mask inside a square of size n*n.
    Used in images preprocessing to hide background

    :return: n*n array
    """
    y, x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y/2 <= r*r/2
    return mask


def preprocess_face(gray, mask, fa, face=None):
    """
    Preprocess a face in a greyscale image.

    Operations:
     - align face (if :face: is not None), which
      return a cropped face
     - equalize the image's histogram
     - hide the background

    :gray: grayscale image
    :mask: mask that will be used to hide background
    :fa: FaceAligner object from imutils.face_utils
    :face: a face found by a dlib detector. If None, no
           alignment is performed

    :return: grayscale image
    """
    if face is not None:
        gray = fa.align(gray, gray, face)
    else:
        gray = cv.resize(gray, (224, 224))
    gray = cv.equalizeHist(gray)
    gray[~mask] = 0
    return gray


def preprocess_img(img, mask, new_width, detector, fa, skip=False):
    """
    Preprocess a BGR image containing a face (if there are more
    faces, only one is taken into account)

    :img: a BGR image
    :mask: mask that is used to hide the background
    :detector: dlib face detector
    :fa: FaceAligner object from imutils.face_utils
    :new_width: the picture is resized to match this width
                (the smaller the image, the faster the detector)
    :skip: return None if no face is detected (insted of preprocessing the
           picture with the function preprocess_face anyway)

    :return: grayscale picture where the face is aligned,
              the histogram equalized, the background hidden by the mask
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=new_width)
    faces = detector(gray, 2)

    # # Old version, saving because who knows
    # if len(faces) > 0:
    #     # if a face is detected, align it
    #     face = faces[0]
    #     # (x, y, w, h) = rect_to_bb(face)
    #     gray = fa.align(gray, gray, face)
    # else:
    #     # otherwise, we know it's there, so
    #     # just resize it
    #     gray = cv.resize(gray, (224, 224))
    # # equalize histogram
    # gray = cv.equalizeHist(gray)

    # # apply mask
    # gray[~mask] = 0

    face = None
    if len(faces) > 0:
        face = faces[0]
    elif skip:
        return None

    gray = preprocess_face(gray, mask, fa, face)
    return gray


def pictures_train_test_val_split(input_dir='pictures/',
                                  output_dir='pictures2/',
                                  val_ratio=0.2, test_ratio=0.1,
                                  classes=None, numpy_random_seed=1,
                                  equal_size=True, verbose=False):
    """
    Split picture classes/folders contained inside :input_dir: in
    train/val/test folders contained in :output_dir:

    :input_dir: a folder containing pictures split in different classes
    :output_dir: folder that will contain the folders split into further
                 train/val/test/folders
    :val_ratio: size of the validation set (between 0 and 1)
    :test_ratio: size of the test set (between 0 and 1)
    :classes: specify which classes to take into account (if None, all 
              classes inside :input_dir: are considered)
    :numpy_random_seed: seed for numpy random number generator
    :equal_size: if True, each class will have train/val/test folders
                 of equal size
    :verbose: print some information during execution

    """
    if classes is None:
        classes = os.listdir(input_dir)

    np.random.seed(numpy_random_seed)

    min_dim = min([len(os.listdir(input_dir+"/"+cls))
                   for cls in os.listdir(input_dir)])

    for cls in classes:
        os.makedirs(output_dir + '/train/' + cls, exist_ok=True)
        os.makedirs(output_dir + '/val/' + cls, exist_ok=True)
        os.makedirs(output_dir + '/test/' + cls, exist_ok=True)

        src = input_dir + "/" + cls

        image_extensions = ["jpg", "png", "jpeg"]
        allFileNames = os.listdir(src)
        allFileNames = [filename for filename in allFileNames if
                        filename.split(".")[-1] in image_extensions]

        np.random.shuffle(allFileNames)

        if equal_size:
            allFileNames = allFileNames[:min_dim]

        trainFileNames, valFileNames, testFileNames = \
            np.split(
                np.array(allFileNames),
                [int(len(allFileNames) * (1 - val_ratio - test_ratio)),
                 int(len(allFileNames) * (1 - test_ratio))])

        trainFileNames = \
            [src+'/' + name for name in trainFileNames.tolist()]

        valFileNames = \
            [src+'/' + name for name in valFileNames.tolist()]

        testFileNames = \
            [src+'/' + name for name in testFileNames.tolist()]

        if verbose:
            print(f"Current class: {cls}")
            print(f'Total images: {len(allFileNames)}')
            print(f'Training: {len(trainFileNames)}')
            print(f'Validation: {len(valFileNames)}')
            print(f'Testing: {len(testFileNames)}\n')

        for name in trainFileNames:
            shutil.copy(name, output_dir + '/train/' + cls)

        for name in valFileNames:
            shutil.copy(name, output_dir + '/val/' + cls)

        for name in testFileNames:
            shutil.copy(name, output_dir + '/test/' + cls)


def get_left_eye(shape, img):
    """
    Get left eye from an image

    :shape: shape returned by dlib predictor
    :img: image containing the eye from which the shape was obtained

    :return: array containing the left eye
    """

    x1 = shape.part(36).x
    x2 = shape.part(39).x
    y1 = shape.part(37).y
    y2 = shape.part(40).y

    return img[y1:y2, x1:x2]


def get_right_eye(shape, img):
    """
    Get right eye from an image

    :shape: shape returned by dlib predictor
    :img: image containing the eye from which the shape was obtained

    :return: array containing the right eye
    """

    x3 = shape.part(42).x
    x4 = shape.part(45).x
    y3 = shape.part(43).y
    y4 = shape.part(46).y

    return img[y3:y4, x3:x4]


def get_eyes(shape, img):
    """
    Get eyes from an image

    :shape: shape returned by dlib predictor
    :img: image containing the eyes from which the shape was obtained

    :return: arrays containing the left and right eyes
    """

    left_eye = get_left_eye(shape, img)
    right_eye = get_right_eye(shape, img)

    return left_eye, right_eye

def plot_eyes(img, faces, predictor, gray=None):
    """
    Plot :img:, its grayscale version, and the eyes detected with dlib

    :img: a BGR image
    :faces: faces detected with dlib
    :predictor: dlib predictor
    :gray: the grayscale image used by the dlib predictor. (Can be None)
    """
    if gray is None:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if len(faces) > 0:
        for face in faces:
            shape = predictor(gray, face)
            left_eye, right_eye = get_eyes(shape, img)

            f, axes = plt.subplots(2, 2, figsize=(10, 7))
            axes[0, 0].imshow(img[:, :, ::-1])
            axes[0, 1].imshow(gray, cmap="gray")
            axes[1, 0].imshow(left_eye[:, :, ::-1])
            axes[1, 1].imshow(right_eye[:, :, ::-1])
            plt.show()


def load_model(model_number, models_dir="models"):
    """
    Load model and relative info for Images section

    :model_number: The number of the model
    :models_dir:   The directory containing the models

    :return: the loaded keras model, its class labels and
              the function needed to preprocess pictures before passing
              them to the model
    """
    import keras
    model = keras.models.load_model(f"./{models_dir}/{model_number}_model.h5")
    with open(f"{models_dir}/{model_number}_model.json") as jf:
        json_file = json.load(jf)

    # I didn't save the "class_indices" field for some models...
    if "class_indices" not in json_file:
        if model.output_shape[1] == 8:
            labels = np.array(["alessandro", "alinda", "cami", "gian",
                               "luca", "mamma", "papi", "umbe"])

        if model.output_shape[1] == 7:
            labels = np.array(["alessandro", "alinda", "cami", "gian",
                               "luca", "mamma", "papi"])
        if model.output_shape[1] == 6:
            labels = np.array(["alessandro", "alinda", "cami", "gian",
                               "mamma", "papi"])
        if model.output_shape[1] == 5:
            labels = np.array(["alinda", "cami",  "gian",
                               "mamma",  "papi", ])
    else:
        labels = np.array(json_file["class_indices"])

    if json_file["featurewise_center"] == True:
        mu = np.array([float(e) for e in json_file["mean"]]).reshape(1, 1)
        std = np.array([float(e) for e in json_file["std"]]).reshape(1, 1)
        def preprocess_fun(x): return (x-mu)/std
    else:
        def preprocess_fun(x): return x/255

    return model, labels, preprocess_fun


def find_last_filename_id(filenames):
    """
    Find last filename id from a list of filenames

    An id is, e.g., the number "123" in the filename 
        "some/directory/gian_123.png"

    :filenames: list of filenames

    :return: highest id in :filenames:
    """
    if len(filenames) == 0:
        latest_picture = -1
    else:
        latest_picture = max(
            [int(filenames[i].split(".")[0].split("_")[1]) for i in range(len(filenames))])
    return latest_picture
