import cv2
import imutils
import numpy as np

img = cv2.imread('/home/loc/Workspace/deskew/image_19042023124941.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
H,W = gray.shape
center_img = (W/2,H/2)

gray = cv2.bitwise_not(gray)

# threshold the image, setting all foreground pixels to 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coordinates = np.column_stack(np.where(thresh > 0)) # [n,2]

# calculate the skew angle by minAreaRect method
# of cv2 which returns an angle range from -90 to 0 degrees
# (where 0 is not include)
# the rotated angle of the text region will be stored in the ang variables
ang = cv2.minAreaRect(coordinates)[-1]
print('ang before',ang)

# Add a condition for the angle, if the text region nagle is smaller than -45
# we will add a 90 degrees
# else we will multiply the angle iwth a minus to make the angle positive
if ang < -45:
    ang = -(90 + ang)
else:
    ang = -ang
print('ang after',ang)

M = cv2.getRotationMatrix2D(center_img,ang,1.0)
# rotated = cv2.warpAffine(img,M,(W,H),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
rotated = imutils.rotate_bound(img,ang)
cv2.imwrite('/home/loc/Workspace/deskew/rotated.jpg',rotated)

# cv2.imshow('out',rotated)
#
# k = cv2.waitKey(0)
#
# cv2.destroyAllWindows()
