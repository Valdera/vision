import cv2


def RecoverHE(sceneRadiance):
    for i in range(3):
        sceneRadiance[:, :, i] = cv2.equalizeHist(sceneRadiance[:, :, i])
    return sceneRadiance


def see(img):
    sceneRadiance = RecoverHE(img)
    return sceneRadiance
