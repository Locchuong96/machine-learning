
# Good for storage image in numpy array
# save numpy array as npy file
import cv2
from numpy import savez_compressed
# define data
img = cv2.imread("./cat.png")
# save data to npy file
savez_compressed('data.npy',data)
