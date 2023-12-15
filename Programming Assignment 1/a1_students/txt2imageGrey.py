import matplotlib.pyplot as plt
import numpy as np 
import sys
import os 


filename = sys.argv[1]
file_prefix = os.path.splitext(filename)[0]
data = np.loadtxt(filename)
width = int(data[0])
height = int(data[1])
data = data[2:]

print("avg = ", np.average(data))


img = data.reshape(height, width)
print(img)
plt.imsave(file_prefix + ".png", img, cmap="gray", vmin=0, vmax=255)


