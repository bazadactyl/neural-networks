import os
import cv2
import numpy as np

def _load_X():
    label_dir = 'VOCdevkit/VOC2012/labels/'
    image_dir = 'VOCdevkit/VOC2012/JPEGImages/'
    X = []

    if not os.path.exists(label_dir):
        print("Generate labels before loading dataset.")
        return

    if not os.path.exists(image_dir):
        print("No image directory found.")
        return

    for i in os.listdir(label_dir):
        path = "%s%s.jpg" % (image_dir,
                             os.path.splitext(i)[0])
        X.append(cv2.imread(path, 1))

    return X

def _format_y():
    pass

def _load_y():
    pass

def load_pascal():
    X = _load_X()
    pass

def main():
    load_pascal()

if __name__ == '__main__':
    main()