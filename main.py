import cv2
import numpy as np
#from matplotlib import pyplot as plt
from parameters import *
from crop import *
from showImage import *
import operator

class Contour:
    def __init__(self, color_number,contour):
        self.color_number = color_number
        self.contour = contour
        self.contour_area = cv2.contourArea(contour)
        #TODO: change contour_center formula
        M = cv2.moments(contour)
        if M["m00"] != 0:
            self.contour_center = int(M['m10']/M['m00'])
        else:
            self.contour_center = 0

        x, y, w, h = cv2.boundingRect(contour)
        self.rect_area = w*h
        self.rect_center = x + w//2


def findColors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #=========================CANNY
    edges = cv2.Canny(img, 100, 200)
    cv2.imshow("canny",edges)
    #=========================

    areas = [] #areas = squares that we need to detect (filter)
    for i in range(BLOCK_NUMBER):
        mask = cv2.inRange(hsv, COLORS_MIN[:,i], COLORS_MAX[:,i])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
        # TODO: better color ranges
        # TODO: solve brightness
        if i==0:
            showPicture("yellow", mask)
        if i==1:
            showPicture("green", mask)
        if i == 2:
            showPicture("black HSV", mask)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            mask = cv2.inRange(lab, blackLAB[0], blackLAB[1])
            showPicture("black LAB", mask)
        if i==3:
            showPicture("blue", mask)
        if i == 4:
            showPicture("orange", mask)

        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # TODO: better contour selector
        if len(contours) > 0:
            biggest_contour = max(contours, key=cv2.contourArea)
            areas.append(Contour(i, biggest_contour))
            for j in range(len(contours)):
                drawSquare(img, contours[j], i)
    areas.sort(key=operator.attrgetter("contour_area"),reverse=True)
    if len(areas) >= 3:
        three_largest_areas = [areas[0],areas[1],areas[2]]
        three_largest_areas.sort(key=operator.attrgetter("contour_center"))
        print(checkColorCombination(three_largest_areas))

def equalArrays(arr1,arr2):
    for i in range(3):
        if arr1[i] != arr2[i]:
            return False
    return True

def checkColorCombination(areas):
    colors = np.array([areas[0].color_number,
              areas[1].color_number,
              areas[2].color_number])
    for i in range(CONSTRUCTION_PLANS_NUMBER):
        if np.array_equal(np.array(colors),np.array(CONSTRUCTION_PLANS[i])):
            return colorNumber2color(colors[0]) + ' ' + colorNumber2color(colors[1]) + ' ' + colorNumber2color(colors[2])
    colors = colors[::-1] #invert color position in array
    for i in range(CONSTRUCTION_PLANS_NUMBER):
        if np.array_equal(colors,CONSTRUCTION_PLANS[i]):
            return colorNumber2color(colors[0]) + ' ' + colorNumber2color(colors[1]) + ' ' + colorNumber2color(colors[2])
    return "Combination doesnt exist: " + colorNumber2color(colors[0]) + ' ' + colorNumber2color(colors[1]) + ' ' + colorNumber2color(colors[2])

def start(frame):

    #frame = cv2.GaussianBlur(frame,(15,15),0)
    #frame = cv2.blur(frame,(11,11))

    findColors(frame)
    showPicture('orig', frame)

if VIDEO:
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        start(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    folder = '/home/bici/Desktop/Memristor Slike'
    imgPath = os.path.join(folder,'8.jpg')
    targetPath = os.path.join(folder,'3.jpg')
    frame = cv2.imread(imgPath, 1)

cv2.waitKey(0)
cv2.destroyAllWindows()