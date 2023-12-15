import matplotlib.pyplot as plt
import numpy as np 
import sys
import os 

filename = sys.argv[1]
file_prefix = os.path.splitext(filename)[0]
f = open(filename)

width = int(f.readline())
height = int(f.readline())
print("width", width, "height", height)
# data = data[2:]
data = np.loadtxt(f)
data = data / 255.0

# print(data)


f.close()
img=data.reshape(height, width, 3)

plt.imsave(file_prefix + ".png", img, vmin=0, vmax=255)
# print(data)

# print("avg = ", np.average(data))


# img = data.reshape(height, width)
# print(img)
# plt.imsave(filename + ".png", img, cmap="gray", vmin=0, vmax=255)


