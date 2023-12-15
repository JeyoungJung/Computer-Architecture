import matplotlib.pyplot as plt
import numpy as np 
import sys

filename = sys.argv[1]
img = plt.imread(filename + ".jpg")

# print(img)

r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]
print(r.shape)
width = r.shape[1]
height = r.shape[0]
print(width)

out = open("imageColor.txt", "w")
out.write(str(width) + "\n")
out.write(str(height) + "\n")

for row in img:
    for pixel in row:
        # print(pixel)
        out.write(str(pixel[0]) + " " + str(pixel[1]) + " " + str(pixel[2]) + "\n")

# print(img)
    

out.close()
# print(r)