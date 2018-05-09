import os
import cv2
import numpy as np
import time
from socket import *


SQUARE_COLORS = {
    # TODO: better color ranges
    # https://www.rapidtables.com/web/color/RGB_Color.html
    # name      min             max             color to draw
    'yellow': ([24, 50, 100], [41, 255, 255], (128, 255, 255)),
    'green': ([42, 60, 120], [70, 255, 255], (0, 255, 0)),
    'black': ([0, 0, 0], [180, 255, 120], (0, 0, 0)),
    'blue': ([90, 50, 120], [130, 255, 255], (255, 0, 0)),
    'orange': ([5, 50, 50], [20, 255, 255], (0, 0, 255)),
}

BOUNDARY_COLOR = {
    # For comparison: RAL 7032 = [22.5, 28, 181]
    # 'gray': ([0, 0, 76], [180, 76, 207], (128,128,128))
    # TODO: better gray ranges
    'gray': ([0, 0, 100], [180, 76, 255], (128, 128, 128))
}

CONSTRUCTION_PLANS = {
    '4,2,1': [["orange", "black", "green"], 0],
    '0,2,3': [["yellow", "black", "blue"], 0],
    '3,1,4': [["blue", "green", "orange"], 0],
    '0,1,2': [["yellow", "green", "black"], 0],
    '2,0,4': [["black", "yellow", "orange"], 0],

    '1,0,3': [["green", "yellow", "blue"], 0],
    '3,4,2': [["blue", "orange", "black"], 0],
    '1,4,0': [["green", "orange", "yellow"], 0],
    '2,3,1': [["black", "blue", "green"], 0],
    '4,3,0': [["orange", "blue", "yellow"], 0]
}


def check_contour_area(contour, max_area=700, min_area=300):
    return max_area > cv2.contourArea(contour) > min_area


"""
def is_rectangle(contour, tolerance=0.075):
   tolerance(Image3) < 0.1
       # tolerance(Image22) > 0.05
       approx = cv2.approxPolyDP(contour, tolerance * cv2.arcLength(contour, True), True)
    return len(approx) == 4
"""


def is_perspective_square(contour, max_ratio=2, min_ratio=0.7):
    _, _, w, h = cv2.boundingRect(contour)
    return max_ratio * w >= h >= min_ratio * w


def is_correctly_oriented(contour, max_angle=15):
    # https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
    _, _, angle = cv2.minAreaRect(contour)
    angle = abs(angle)
    if angle < max_angle or abs(angle - 90) < max_angle:
        return True

"""
def get_avg_color_of_neighbor(image_hsv, x, y, dist=7):
    # Check if image boundaries are exceeded
    top = y - dist
    bottom = y + dist
    left = x - dist
    right = x + dist
    if image_hsv.shape[1] <= right:
        right = image_hsv.shape[1] - 1
    if image_hsv.shape[0] <= bottom:
        bottom = image_hsv.shape[0] - 1
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right <= 0 or left >= image_hsv.shape[1] or top >= image_hsv.shape[
        0] or bottom <= 0 or right == left or top == bottom:
        return np.array([-1, -1, -1])

    # Find mean hsv value of region
    neighbor_mean = [image_hsv[top: bottom, left: right, i].mean() for i in range(3)]
    return np.array(list(map(int, neighbor_mean)))


def is_color_gray(color_to_check):
    is_gray = False
    for color, value in BOUNDARY_COLOR.items():
        is_gray = is_gray or ((value[0] < color_to_check).all and (color_to_check < value[1]).all)
    return is_gray


def is_on_wall(image_hsv, contour, euclidian_distance=50, height = 15):
    # Biggest euclidean distande: Image 15 17 20
    # TODO: is_on_wall function VERY SLOW! ~2.5 seconds execution
    x, y, w, h = cv2.boundingRect(contour)

    # Get coordinates above & below the rectangle
    y_down = int(y + h + height)
    y_up = int(y - height)
    x_middle = int(x + w / 2)
    #cv2.imshow('gray', cv2.inRange(image_hsv, np.array(BOUNDARY_COLOR['gray'][0]), np.array(BOUNDARY_COLOR['gray'][1])))

    # Find average color
    # TODO: use wider x area?
    color_up = get_avg_color_of_neighbor(image_hsv, x_middle, y_up)
    color_down = get_avg_color_of_neighbor(image_hsv, x_middle, y_down)

    # Check Euclidean distance between up and down colors, afterwards and if color is gray
    # TODO: Remove is_color_gray? Doesn't help much..
    # Irony? Code is faster with is_color_gray
    dist = np.linalg.norm(color_up - color_down)
    return dist < euclidian_distance and is_color_gray(color_up) and is_color_gray(color_down)
"""

def are_rectangles_adjacent(filtered_rectangles, x_multiplier=1.4, y_multiplier=0.5):
    adjacent_rectangles = {}
    for outer_color, outer_rectangle in filtered_rectangles.items():
        has_adjacent = False
        x, y, w, h = outer_rectangle
        x1 = x - x_multiplier * w
        y1 = y - y_multiplier * w
        x2 = x + w + x_multiplier * h
        y2 = y + h + y_multiplier * h
 
        for inner_color, inner_rectangle in filtered_rectangles.items():
            if inner_color == outer_color:
                continue
            x, y, w, h = inner_rectangle
            if x2 > x + w > x > x1 and y2 > y + h > y > y1:
                has_adjacent = True
 
        if has_adjacent:
            adjacent_rectangles[outer_color] = outer_rectangle
    return(adjacent_rectangles)


def filter_contours(image_hsv, mask):
    filtered_contours = []
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # TODO: reorder functions by execution time
        if check_contour_area(contour) and \
                is_perspective_square(contour) and \
                is_correctly_oriented(contour):
            filtered_contours.append(contour)
    return filtered_contours


def check_color_combination(square_x_color):
    colors = np.array([item[0] for item in square_x_color])
    for key, value in CONSTRUCTION_PLANS.items():
        if np.array_equal(np.array(colors), np.array(value[0])):
            value[1] += 1
            return True
    colors = colors[::-1]  # invert color position in array
    for key, value in CONSTRUCTION_PLANS.items():
        # TODO: is the comma a problem?
        if np.array_equal(colors, value[0]):
            value[1] += 1
            return True
    return False


# def crop_image(image, x_window = 55, y_window = 45, x_offset = 0, y_offset = -60):
def crop_image(image, x_window = 100, y_window = 60, x_offset = 0, y_offset = 0):
    y, x, _ = image.shape
    top = y // 2 - y_window // 2 + y_offset
    bottom = y // 2 + y_window // 2 + y_offset
    left = x // 2 - x_window // 2 + x_offset
    right = x // 2 + x_window // 2 + x_offset
    if top < 0:
        top = 0
    if bottom >= y:
        bottom = y - 1
    if left < 0:
        left = 0
    if right >= x:
        right = x - 1 
    return image[top:bottom, left:right]


def send_to_server(data, host = "192.168.4.1", port = 5000):
    try:
        s=socket(AF_INET, SOCK_STREAM)
        s.connect((host,port))
        s.send(data.encode('utf-8'))
    except:
        pass

def key_with_max_val(square_x_color):
    return max(CONSTRUCTION_PLANS, key=lambda k: CONSTRUCTION_PLANS[k][1])

def sum_of_construction_plans():
    val = 0
    for key, value in CONSTRUCTION_PLANS.items():
        val += value[1]
    return val

def process_image(image):
    global not_sent_anything
    global data_to_send
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_with_squares = image.copy()

    square_masks = {}
    global_square_mask = 0
    filtered_mask = np.zeros(image.shape)
    square_x_color = {}
    for color, value in SQUARE_COLORS.items():
        mask = cv2.inRange(image_hsv, np.array(value[0]), np.array(value[1]))
        kernel = np.ones((7,7),np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        square_masks[color] = mask
        global_square_mask = cv2.bitwise_or(global_square_mask, mask)

        # Get contours
        contours = filter_contours(image_hsv, mask)
        #print(color + ": " + str(len(contours)))

        # Display contours
        for contour in contours:
            square_x_color[color] = cv2.boundingRect(contour)
            cv2.drawContours(image_with_squares, [contour], -1, value[2], -1)
            cv2.drawContours(image_with_squares, [contour], -1, (255, 255, 255), 2)

    # Get adjacent contours
    square_x_color = are_rectangles_adjacent(square_x_color)

    # Find order
    import operator
    square_x_color = sorted(square_x_color.items(), key=operator.itemgetter(1))


    if check_color_combination(square_x_color):
        print("COMBINATION EXISTS <===========> " + str([col[0] for col in square_x_color]))
    else:
        print("Combination doesn't exist" + str([col[0] for col in square_x_color]))
    # Check if color combination exists
    if DEBUG_MODE == True:
        cv2.imshow("global_square_mask", global_square_mask)
        cv2.imshow("image_with_squares", image_with_squares)
        for color,masked in square_masks.items():
            cv2.imshow(color,masked)

    #send_to_server(key_with_max_val(CONSTRUCTION_PLANS)) 
    #if CONSTRUCTION_PLANS[key_with_max_val(CONSTRUCTION_PLANS)][1]:
    #    send_to_server(key_with_max_val(CONSTRUCTION_PLANS))
    data_to_send = key_with_max_val(CONSTRUCTION_PLANS) 
    send_to_server(data_to_send)
    print(data_to_send)
  	
""" BRKICEVA ZELJA
    if (sum_of_construction_plans() >= TIMES_COMBINATION_FOUND) and not_sent_anything:
        not_sent_anything = False
        data_to_send = key_with_max_val(CONSTRUCTION_PLANS)
        send_to_server(data_to_send) 
        print(data_to_send) 	 
    if not_sent_anything == False:
        send_to_server(data_to_send) 
        print(data_to_send)
 """

#=================================== IMPORTANT PARAMETERS
SEND_TIME = 120
DEBUG_MODE = True
TIMES_COMBINATION_FOUND = 15
#===================================

started_measuring_time = False
PERIOD_OF_TIME = 1000  # in seconds (10 minutes)
start_time = time.time()
global not_sent_anything
not_sent_anything = True 
global data_to_send
data_to_send = ""

started_measuring_time = False
time_exceeded = False
PERIOD_OF_TIME = 600  # in seconds (10 minutes)
start_time = time.time()

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    curr_time = time.time()

    #if DEBUG_MODE:
    #    cv2.imshow("orig",image)
    image = crop_image(image)
    if DEBUG_MODE:
        cv2.imshow("cropped",image)

    # find the order of colors
    process_image(image)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or time.time() > start_time + PERIOD_OF_TIME:
        break

cv2.destroyAllWindows()

if DEBUG_MODE:
    print(CONSTRUCTION_PLANS)
    print("MAXVAL = ", CONSTRUCTION_PLANS[key_with_max_val(CONSTRUCTION_PLANS)])



