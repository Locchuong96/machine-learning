# load numpy array from npz file
from numpy import load
# load dict of arrays
dict_data = load('data.npz')
# extract the first_array
data = dict_data['arr_0']
#print the array
print(data)
