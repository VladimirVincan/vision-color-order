import cv2
from matplotlib import pyplot as plt
from showImage import *
from parameters import *
import _tkinter

#slicno kao histogram_backprojection

def butterworthFilter(data):
    from scipy import signal
    # First, design the Buterworth filter
    N = 1  # Filter order
    Wn = 0.4  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')

    # Second, apply the filter
    return signal.filtfilt(B, A, data, axis=0)

def histogram(img, bw = 0.5, butterworth = False, range_type="relative"):
    color = ('b', 'g', 'r')
    range = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        if butterworth:
            histr = butterworthFilter(histr)
        maxPos = np.argmax(histr)
        l = r = maxPos
        if range_type=="relative":
            while histr[r] >= bw*histr[maxPos] and r < 255:
                r+=1
            while histr[l] >= bw*histr[maxPos] and l > 0:
                l-=1
        elif range_type=="absolute":
            l = r = bw
            if l<0:
                l=0
            if r>255:
                r=255
        #print(histr[l])
        range.append([l,r])

        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    return range

class CropImage:
    def __init__(self):
        self.roi = []
        self.drag = False

    def setColor(self, color_number):
        self.color = contourColor(color_number)

    def on_mouse(self, event, x, y, flags, img):
        if event == cv2.EVENT_LBUTTONDOWN and not self.drag:
            #print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            self.roi = [x,y,x,y]
            self.drag= True

        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.roi[2] = x
            self.roi[3] = y
            imgClone = img.copy()
            cv2.rectangle(imgClone, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), self.color, 3)
            showPicture("img", imgClone)

        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            #print('End Mouse Position: ' + str(x) + ', ' + str(y))
            self.roi[2] = x
            self.roi[3] = y
            self.drag = False

def cropSquares(img):
    cropImage = CropImage()
    cv2.setMouseCallback('img', cropImage.on_mouse, img)
    cropped_images = []
    i = 0
    while (i < BLOCK_NUMBER):
        cropImage.setColor(i)
        pressedKey = chr(cv2.waitKey(0) & 255)
        if 's' == pressedKey: #save square
            cropped_images.append(img[cropImage.roi[1]:cropImage.roi[3], cropImage.roi[0]:cropImage.roi[2]])
            i += 1
        elif 'n' == pressedKey: #next
            i += 1
        elif 'r' == pressedKey: #restart
            i = 0
        elif 'q' == pressedKey: #quit
            cv2.destroyAllWindows()

    return cropped_images

def updateRange(img):
    rangeBlack = histogram(crop)
    COLORS_MIN[:, 2] = np.array(rangeBlack)[:, 0]
    COLORS_MAX[:, 2] = np.array(rangeBlack)[:, 1]
