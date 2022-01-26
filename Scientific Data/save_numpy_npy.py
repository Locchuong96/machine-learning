# Good for storage image in numpy array
# save numpy array as npy file
import cv2
from numpy import save
# define data
img = cv2.imread("./cat.png")
# save data to npy file
save('data.npy',data)
