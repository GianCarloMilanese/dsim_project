#!/usr/bin/env python3

import os
import numpy as np
import shutil

def pictures_train_test_val_split(input_dir='pictures/',
                                  output_dir='pictures2/',
                                  val_ratio=0.2, test_ratio=0.1,
                                  classes=None, numpy_random_seed=1,
                                  equal_size = True, verbose=False):

    if classes is None:
        classes = os.listdir(input_dir)

    np.random.seed(numpy_random_seed)

    min_dim = min([len(os.listdir(input_dir+"/"+cls)) for cls in os.listdir(input_dir)])

    for cls in classes:
        os.makedirs(output_dir + '/train/' + cls, exist_ok=True)
        os.makedirs(output_dir + '/val/' + cls, exist_ok=True)
        os.makedirs(output_dir + '/test/' + cls, exist_ok=True)

        src = input_dir + "/" + cls

        image_extensions = ["jpg", "png", "jpeg"]
        allFileNames = os.listdir(src)
        allFileNames = [filename for filename in allFileNames if \
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

if __name__ == "__main__":
    
    pictures_train_test_val_split("pictures_new", "pictures_split", classes = ["gian", "alinda", "khaled", "luca"], verbose=True)
