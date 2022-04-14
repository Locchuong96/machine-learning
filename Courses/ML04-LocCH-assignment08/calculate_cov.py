import os 
import glob
import numpy as np 
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

IMG_PATH  = './CroppedYale'
dims = (192,168)

# Create function read a image name and return the value index, also the label of that image
def check_index(image,subjects):
    a = None
    for i in subjects:
        if i in image:
            a = subjects.index(i)
            break
        else:
            pass
    return a

# Get all label
print("Getting all labels")
subjects = os.listdir(IMG_PATH)
print("Done get all labels \n")

# Get all images
print("Getting all images...")

images = []
floders = os.listdir(IMG_PATH)
for f in tqdm(floders):
	images += os.listdir(os.path.join(IMG_PATH,f))

print("Done getting all images! \n")

# Get all available images
print("Getting all available images...")

images_pgm = []
# take all pgm
for i in tqdm(images):
	if '.pgm' in i:
		images_pgm.append(i)

for i in tqdm(images_pgm):
	if '.bad' in i or 'Ambient' in i or len(i) != 24:
		images_pgm.remove(i)

print("Done get all availale images! \n")

# Decode label to numpy array number
print("Decoding labels...")
Y = [check_index(x,subjects) for x in tqdm(images_pgm)]
Y = np.array(Y)
print("Done decoding label to number! \n")

# Covert all available images to array
X = np.zeros((len(images_pgm),dims[0]*dims[1])) # ravel image

for i in tqdm(range(len(images_pgm))):

	image = Image.open( os.path.join(IMG_PATH,images_pgm[i].split('_')[0],images_pgm[i]))

	if (image.height,image.width) != dims:
		image = image.resize(dims)
	
	X[i] = np.array(image.getdata())
    
print("X shape: " + str(X.shape))
print("Done  convert all available image to numpy!")

# Calculate mean
print("Calculating mean...")
mean = np.mean(X,axis =0)
X_dash = X - mean
print(f"X_dash.shape: {X_dash.shape}")

X_dash = X_dash.astype('float16')
X = X.astype('float16')

print('Calculating covariance matrix...')
cov  = 1/len(X) * X_dash.T @ X_dash
print("Saving your matrix")
np.save('cov.npy',cov)
print("Done!!!")

