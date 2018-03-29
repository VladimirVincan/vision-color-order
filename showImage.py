import cv2
import numpy as np

def contourColor(color_number):
    if color_number == 0:
        contour_color = (128, 255, 255)
    elif color_number == 1:
        contour_color = (0, 255, 0)
    elif color_number == 2:
        contour_color = (0, 0, 0)
    elif color_number == 3:
        contour_color = (255, 0, 0)
    else:
        contour_color = (0, 0, 255)
    return contour_color

def colorNumber2color(color_number):
    if color_number == 0:
        return "yellow"
    if color_number == 1:
        return "green"
    if color_number == 2:
        return "black"
    if color_number == 3:
        return "blue"
    return "orange"

def color2colorNumber(color):
    if color == "yellow":
        return 0
    if color == "green":
        return 1
    if color == "black":
        return 2
    if color == "blue":
        return 3
    return 4

def constructionPlans2colorNumber(colors):
    return [color2colorNumber(colors[0]),
            color2colorNumber(colors[1]),
            color2colorNumber(colors[2])]

def showPicture(name, img, ratio = 1):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    #print(name, img.shape[1], img.shape[0])
    cv2.resizeWindow(name, int(img.shape[1]*ratio), int(img.shape[0]*ratio))

def selectROI(im):
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI("Image", im, fromCenter, showCrosshair)
    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)

def drawSquare(img, contour, color_number):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], -1, contourColor(color_number), 5)
