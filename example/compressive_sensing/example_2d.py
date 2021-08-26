import numpy as np
from cv2 import cv2 as cv

from example.compressive_sensing.sensing import create_sensing_matrix

im = cv.imread("example.jpg")
shape = im.shape

im = im.flatten()

N = len(im)
D = int(0.3 * N)

print(D, N)
sensing_matrix = create_sensing_matrix(D, N)

measure_im = sensing_matrix @ im

draw_im = sensing_matrix.T @ measure_im

cv.imshow("figure", draw_im.reshape(shape))
cv.waitKey(0)
cv.DestroyAllWindows()
