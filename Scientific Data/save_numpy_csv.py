# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
l = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]
data = asarray([l])
savetxt('data.csv',data,delimiter = ',')
