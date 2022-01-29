from skimage.color import rgb2hsv, hsv2rgb
import cv2
import numpy as np

#
# # ----------------------------RGHS--------------------------------
# def cal_equalisation(array, a,height,width):
#     Array = array * a
#     for i in range(height):
#         for j in range(width):
#             if(Array[i][j]>255):
#                 Array[i][j] = 255
#             elif(Array[i][j]<0):
#                 Array[i][j] = 0
#             else:
#                 pass
#     return Array
#
# def RGB_equalisation(r, g, b,height,width):
#     float_r = r.astype(np.float64)
#     float_g = g.astype(np.float64)
#     float_b = b.astype(np.float64)
#     r_avg  = np.mean(float_r)
#     g_avg  = np.mean(float_g)
#     b_avg  = np.mean(float_b)
#     a_r = 128/r_avg
#     a_g = 128/g_avg
#     a_b = 128/b_avg
#     # float_r = cal_equalisation(float_r, a_r, height, width)
#     # float_g = cal_equalisation(float_g,a_g,height,width)
#     # float_b = cal_equalisation(float_b,a_b,height,width)
#     return float_r,float_g, float_b
# ----------------------------------------------------------------------

# ---------------------------------UCM----------------------------------


def cal_equalisation(img, ratio):
    Array = img * ratio
    Array = np.clip(Array, 0, 255)
    return Array


def RGB_equalisation(img):
    img = np.float32(img)
    avg_RGB = []
    for i in range(3):
        avg = np.mean(img[:, :, i])
        avg_RGB.append(avg)
    a_r = avg_RGB[0]/avg_RGB[2]
    a_g = avg_RGB[0]/avg_RGB[1]
    ratio = [0, a_g, a_r]
    for i in range(1, 3):
        img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])
    return img
# ----------------------------------------------------------------------


def global_stretching(img_L, height, width):
    I_min = np.min(img_L)
    I_max = np.max(img_L)
    I_mean = np.mean(img_L)
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            p_out = (img_L[i][j] - I_min) * ((1) / (I_max - I_min))
            array_Global_histogram_stretching_L[i][j] = p_out

    return array_Global_histogram_stretching_L


def histogram_r(r_array, height, width):
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(r_array[i][j])
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])
    array_Global_histogram_stretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if r_array[i][j] < I_min:
                # p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = I_min
            elif (r_array[i][j] > I_max):
                p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = 255
            else:
                p_out = int((r_array[i][j] - I_min) *
                            ((255 - I_min) / (I_max - I_min))) + I_min
                array_Global_histogram_stretching[i][j] = p_out
    return (array_Global_histogram_stretching)


def histogram_g(r_array, height, width):
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(r_array[i][j])
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])
    array_Global_histogram_stretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if r_array[i][j] < I_min:
                p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = 0
            elif (r_array[i][j] > I_max):
                p_out = r_array[i][j]
                array_Global_histogram_stretching[i][j] = 255
            else:
                p_out = int((r_array[i][j] - I_min) *
                            ((255) / (I_max - I_min)))
                array_Global_histogram_stretching[i][j] = p_out
    return (array_Global_histogram_stretching)


def histogram_b(r_array, height, width):
    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            R_rray.append(r_array[i][j])
    R_rray.sort()
    I_min = int(R_rray[int(length / 500)])
    I_max = int(R_rray[-int(length / 500)])
    array_Global_histogram_stretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if r_array[i][j] < I_min:
                array_Global_histogram_stretching[i][j] = 0
            elif (r_array[i][j] > I_max):
                array_Global_histogram_stretching[i][j] = I_max
            else:
                p_out = int((r_array[i][j] - I_min) *
                            ((I_max) / (I_max - I_min)))
                array_Global_histogram_stretching[i][j] = p_out
    return (array_Global_histogram_stretching)


def stretching(img):
    height = len(img)
    width = len(img[0])
    img[:, :, 2] = histogram_r(img[:, :, 2], height, width)
    img[:, :, 1] = histogram_g(img[:, :, 1], height, width)
    img[:, :, 0] = histogram_b(img[:, :, 0], height, width)
    return img


def HSVStretching(sceneRadiance):
    sceneRadiance = np.uint8(sceneRadiance)
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = global_stretching(s, height, width)
    img_v_stretching = global_stretching(v, height, width)
    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255
    return img_rgb


def sceneRadianceRGB(sceneRadiance):
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def ucm(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sceneRadiance = RGB_equalisation(img)
    sceneRadiance = stretching(sceneRadiance)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    sceneRadiance = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return sceneRadiance
