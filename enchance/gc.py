import cv2
import numpy as np


def RecoverGC(sceneRadiance):
    sceneRadiance = sceneRadiance/255.0
    for i in range(3):
        sceneRadiance[:, :, i] = np.power(
            sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 0.7)
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def see(img):
    sceneRadiance = RecoverGC(img)
    return sceneRadiance
