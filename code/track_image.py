import matplotlib.pyplot as plt
import trackpy as tp
import cv2
import numpy as np

# load the image
image = cv2.imread('D:/Exp_Soft_Matter/images/Exercise0000.jpg')
gray = image[:,:,0]

# blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0) # for particles

# detect circles
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=50,
    param2=10,
    minRadius=35,
    maxRadius=35
)

# draw circles into image
if circles is not None:
    print(f'number of circles: {len(circles[0,:])}')
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw center
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
    print(f'number of circles: 0')


# resize image
display_width = 1000
display_height = 1000
resized_image = cv2.resize(image, (display_width, display_height))
resized_blur = cv2.resize(blurred, (display_width, display_height))

cv2.imshow('Circles', resized_image)
cv2.imshow('BLURRED', resized_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
