import numpy as np
import cv2
import os.path

VIDEO = True

SATURATION_MIN = 100
COLORS_MIN = np.array([
    [21,SATURATION_MIN,100], #yellow
    [40,SATURATION_MIN,50], #green
    [0,0,0],      #black
    [100,SATURATION_MIN,100],#blue
    [10,SATURATION_MIN,100]  #orange
])

COLORS_MAX = np.array([
    [36,255,255], #yellow
    [70,255,255], #green
    [180,255,40], #black
    [130,255,255],#blue
    [20,255,255]  #orange
])

blackLAB = np.array([
    [0, 0, 0],
    [30,255,255]
])

COLORS_MIN = COLORS_MIN.T
COLORS_MAX = COLORS_MAX.T



CONSTRUCTION_PLANS_NUMBER=10
CONSTRUCTION_PLANS = np.matrix([
    ["orange","black","green"],
    ["yellow","black","blue"],
    ["blue","green","orange"],
    ["yellow","green","black"],
    ["black","yellow","orange"],

    ["green","yellow","blue"],
    ["blue","orange","black"],
    ["green","orange","yellow"],
    ["black","blue","green"],
    ["orange","blue","yellow"]
])
CONSTRUCTION_PLANS = np.array([
    [4,2,1],
    [0,2,3],
    [3,1,4],
    [0,1,2],
    [2,0,4],

    [1,0,3],
    [3,4,2],
    [1,4,0],
    [0,3,1],
    [4,3,0]
])

BLOCK_NUMBER=5
#BRICK TYPES___________RGB VALUES
yellow  = np.array([0,181,247])     #straw
green   = np.array([59,153,97])      #vegetable
black   = np.array([16,14,14])      #industrial
blue    = np.array([176,124,0])    #solarPanel
orange  = np.array([40,93,208])      #brick



#BRICK TYPES___________HUE VALUES
yellow  = np.array([44,100,97])     #straw
green   = np.array([96,61,60])      #vegetable
black   = np.array([240,13,6])      #industrial
blue    = np.array([198,100,69])    #solarPanel
orange  = np.array([19,81,82])      #brick
white   = np.array([41,67,95])      #golden cube
COLORS = np.column_stack((yellow,green,black,blue,orange))
multiplyVector  = np.array([1 / 2, 2.55, 2.55]) #CONVERT INTO CV2 HSV VALUES
COLORS = np.matrix.round((COLORS.T * multiplyVector).T)