'''
remove the line certifi @ file:///croot/certifi_1671487769961/work/certifi
in exported list of installed packages from conda env
'''
import argparse

# init parser
parser = argparse.ArgumentParser(description = 'Use Kmeans to cluster color in the image')
# add argument to parser
parser.add_argument('-p','--path',type = str, help = 'directory to text file', required = True)
# create arguments
args = parser.parse_args()

path = args.path
file = path
print("Reading ",path)

# read packages.txt text file
with open(path,'r') as f:
    texts = f.readlines()
#print(content)

if __name__ == "__main__":
    
    lines = [] # filtered content
    
    # loop over each line
    for t in texts:
        if 'certifi' in t:
            print("Remove line ",t)
        else:
            lines.append(t)
    
    # write file
    f = open(file,'w')
    for l in lines:
        f.write(l)
    f.close()
    